use super::*;
use polars_core::utils::{accumulate_dataframes_vertical, split_df};
use polars_core::POOL;
use rayon::prelude::*;

/// Take an input Executor and a multiple expressions
pub struct PartitionGroupByExec {
    input: Box<dyn Executor>,
    keys: Vec<Arc<dyn PhysicalExpr>>,
    phys_aggs: Vec<Arc<dyn PhysicalExpr>>,
    maintain_order: bool,
    slice: Option<(i64, usize)>,
    input_schema: SchemaRef,
}

impl PartitionGroupByExec {
    pub(crate) fn new(
        input: Box<dyn Executor>,
        keys: Vec<Arc<dyn PhysicalExpr>>,
        phys_aggs: Vec<Arc<dyn PhysicalExpr>>,
        maintain_order: bool,
        slice: Option<(i64, usize)>,
        input_schema: SchemaRef,
    ) -> Self {
        Self {
            input,
            keys,
            phys_aggs,
            maintain_order,
            slice,
            input_schema,
        }
    }

    fn keys(&self, df: &DataFrame, state: &ExecutionState) -> Result<Vec<Series>> {
        self.keys.iter().map(|s| s.evaluate(df, state)).collect()
    }
}

fn run_partitions(
    df: &DataFrame,
    exec: &PartitionGroupByExec,
    state: &ExecutionState,
    n_threads: usize,
    maintain_order: bool,
) -> Result<Vec<DataFrame>> {
    // We do a partitioned groupby.
    // Meaning that we first do the groupby operation arbitrarily
    // split on several threads. Than the final result we apply the same groupby again.
    let dfs = split_df(df, n_threads)?;

    POOL.install(|| {
        dfs.into_par_iter()
            .map(|df| {
                let keys = exec.keys(&df, state)?;
                let phys_aggs = &exec.phys_aggs;
                let gb = df.groupby_with_series(keys, false, maintain_order)?;
                let groups = gb.get_groups();

                let mut columns = gb.keys();
                // don't naively call par_iter here, it will segfault in rayon
                // if you do, throw it on the POOL threadpool.
                let agg_columns = phys_aggs
                    .iter()
                    .map(|expr| {
                        let agg_expr = expr.as_partitioned_aggregator().unwrap();
                        let agg = agg_expr.evaluate_partitioned(&df, groups, state)?;
                        if agg.len() != groups.len() {
                            Err(PolarsError::ComputeError(
                                format!("returned aggregation is a different length: {} than the group lengths: {}",
                                        agg.len(),
                                        groups.len()).into()
                            ))
                        } else {
                            Ok(agg)
                        }
                    }).collect::<Result<Vec<_>>>()?;

                columns.extend_from_slice(&agg_columns);

                DataFrame::new(columns)
            })
    }).collect()
}

fn estimate_unique_count(keys: &[Series], mut sample_size: usize) -> usize {
    // https://stats.stackexchange.com/a/19090/147321
    // estimated unique size
    // u + ui / m (s - m)
    // s: set_size
    // m: sample_size
    // u: total unique groups counted in sample
    // ui: groups with single unique value counted in sample
    let set_size = keys[0].len();
    let offset = (keys[0].len() / 2) as i64;
    if set_size < sample_size {
        sample_size = set_size;
    }

    let finish = |groups: &GroupsProxy| {
        let u = groups.len() as f32;
        let ui = if groups.len() == sample_size {
            u
        } else {
            groups.idx_ref().iter().filter(|g| g.1.len() == 1).count() as f32
        };

        (u + (ui / sample_size as f32) * (set_size - sample_size) as f32) as usize
    };

    if keys.len() == 1 {
        // we sample as that will work also with sorted data.
        // not that sampling without replacement is very very expensive. don't do that.
        let s = keys[0].sample_n(sample_size, true, false, None).unwrap();
        // fast multi-threaded way to get unique.
        let groups = s.group_tuples(true, false);
        finish(&groups)
    } else {
        let keys = keys
            .iter()
            .map(|s| s.slice(offset, sample_size))
            .collect::<Vec<_>>();
        let df = DataFrame::new_no_checks(keys);
        let names = df.get_column_names();
        let gb = df.groupby(names).unwrap();
        finish(gb.get_groups())
    }
}

// Checks if we should run normal or default aggregation
// by sampling data.
fn can_run_partitioned(keys: &[Series], original_df: &DataFrame, state: &ExecutionState) -> bool {
    if std::env::var("POLARS_NO_PARTITION").is_ok() {
        if state.verbose {
            eprintln!("POLARS_NO_PARTITION set: running default HASH AGGREGATION")
        }
        false
    } else if std::env::var("POLARS_FORCE_PARTITION").is_ok() {
        if state.verbose {
            eprintln!("POLARS_FORCE_PARTITION set: running partitioned HASH AGGREGATION")
        }
        true
    } else if original_df.height() < 1000 && !cfg!(test) {
        if state.verbose {
            eprintln!("DATAFRAME < 1000 rows: running default HASH AGGREGATION")
        }
        false
    } else {
        // below this boundary we assume the partitioned groupby will be faster
        let unique_count_boundary = std::env::var("POLARS_PARTITION_UNIQUE_COUNT")
            .map(|s| s.parse::<usize>().unwrap())
            .unwrap_or(1000);

        let (unique_estimate, sampled_method) = match (keys.len(), keys[0].dtype()) {
            #[cfg(feature = "dtype-categorical")]
            (1, DataType::Categorical(Some(rev_map))) => (rev_map.len(), "known"),
            _ => {
                let sample_frac = std::env::var("POLARS_PARTITION_SAMPLE_FRAC")
                    .map(|s| s.parse::<f32>().unwrap())
                    .unwrap_or(0.001);
                let sample_size = (original_df.height() as f32 * sample_frac) as usize;

                // we never sample more than 1k data points
                let sample_size = std::cmp::min(sample_size, 1_000);
                // we never sample less than 100 data points.
                let sample_size = std::cmp::max(100, sample_size);
                (estimate_unique_count(keys, sample_size), "estimated")
            }
        };
        if state.verbose {
            eprintln!("{} unique values: {}", sampled_method, unique_estimate);
        }

        if unique_estimate > unique_count_boundary {
            if state.verbose {
                eprintln!("estimated unique count: {} exceeded the boundary: {}, running default HASH AGGREGATION",unique_estimate, unique_count_boundary)
            }
            false
        } else {
            true
        }
    }
}

impl Executor for PartitionGroupByExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let dfs = {
            let original_df = self.input.execute(state)?;

            // already get the keys. This is the very last minute decision which groupby method we choose.
            // If the column is a categorical, we know the number of groups we have and can decide to continue
            // partitioned or go for the standard groupby. The partitioned is likely to be faster on a small number
            // of groups.
            let keys = self.keys(&original_df, state)?;

            if !can_run_partitioned(&keys, &original_df, state) {
                return groupby_helper(
                    original_df,
                    keys,
                    &self.phys_aggs,
                    None,
                    state,
                    false,
                    self.slice,
                );
            }
            if state.verbose {
                eprintln!("run PARTITIONED HASH AGGREGATION")
            }

            // Run the partitioned aggregations
            let n_threads = POOL.current_num_threads();

            // set it here, because `self.input.execute` will clear the schema cache.
            state.set_schema(self.input_schema.clone());
            run_partitions(&original_df, self, state, n_threads, self.maintain_order)?
        };
        state.clear_schema_cache();

        // MERGE phase
        // merge and hash aggregate again
        let df = accumulate_dataframes_vertical(dfs)?;
        // the partitioned groupby has added columns so we must update the schema.
        let keys = self.keys(&df, state)?;

        // first get mutable access and optionally sort
        let gb = df.groupby_with_series(keys, true, self.maintain_order)?;
        let mut groups = gb.get_groups();

        #[allow(unused_assignments)]
        // it is unused because we only use it to keep the lifetime of sliced_group valid
        let mut sliced_groups = None;

        if let Some((offset, len)) = self.slice {
            sliced_groups = Some(groups.slice(offset, len));
            groups = sliced_groups.as_deref().unwrap();
        }

        let get_columns = || gb.keys_sliced(self.slice);
        let get_agg = || {
            let out: Result<Vec<_>> = self
                .phys_aggs
                .par_iter()
                // we slice the keys off and finalize every aggregation
                .zip(&df.get_columns()[self.keys.len()..])
                .map(|(expr, partitioned_s)| {
                    let agg_expr = expr.as_partitioned_aggregator().unwrap();
                    agg_expr.finalize(partitioned_s.clone(), groups, state)
                })
                .collect();

            out
        };
        let (mut columns, agg_columns): (Vec<_>, _) =
            POOL.install(|| rayon::join(get_columns, get_agg));

        columns.extend(agg_columns?);
        state.clear_schema_cache();

        Ok(DataFrame::new(columns).unwrap())
    }
}
