use polars_core::series::IsSorted;
use polars_core::utils::{accumulate_dataframes_vertical, split_df};
use rayon::prelude::*;

use super::*;

/// Take an input Executor and a multiple expressions
pub struct PartitionGroupByExec {
    input: Box<dyn Executor>,
    phys_keys: Vec<Arc<dyn PhysicalExpr>>,
    phys_aggs: Vec<Arc<dyn PhysicalExpr>>,
    maintain_order: bool,
    slice: Option<(i64, usize)>,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
    from_partitioned_ds: bool,
    #[allow(dead_code)]
    keys: Vec<Expr>,
    #[allow(dead_code)]
    aggs: Vec<Expr>,
}

impl PartitionGroupByExec {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input: Box<dyn Executor>,
        phys_keys: Vec<Arc<dyn PhysicalExpr>>,
        phys_aggs: Vec<Arc<dyn PhysicalExpr>>,
        maintain_order: bool,
        slice: Option<(i64, usize)>,
        input_schema: SchemaRef,
        output_schema: SchemaRef,
        from_partitioned_ds: bool,
        keys: Vec<Expr>,
        aggs: Vec<Expr>,
    ) -> Self {
        Self {
            input,
            phys_keys,
            phys_aggs,
            maintain_order,
            slice,
            input_schema,
            output_schema,
            from_partitioned_ds,
            keys,
            aggs,
        }
    }

    fn keys(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Vec<Series>> {
        compute_keys(&self.phys_keys, df, state)
    }
}

fn compute_keys(
    keys: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    state: &ExecutionState,
) -> PolarsResult<Vec<Series>> {
    keys.iter().map(|s| s.evaluate(df, state)).collect()
}

fn run_partitions(
    df: &mut DataFrame,
    exec: &PartitionGroupByExec,
    state: &ExecutionState,
    n_threads: usize,
    maintain_order: bool,
) -> PolarsResult<(Vec<DataFrame>, Vec<Vec<Series>>)> {
    // We do a partitioned group_by.
    // Meaning that we first do the group_by operation arbitrarily
    // split on several threads. Than the final result we apply the same group_by again.
    let dfs = split_df(df, n_threads, true);

    let phys_aggs = &exec.phys_aggs;
    let keys = &exec.phys_keys;

    let mut keys = DataFrame::from_iter(compute_keys(keys, df, state)?);
    let splitted_keys = split_df(&mut keys, n_threads, true);

    POOL.install(|| {
        dfs.into_par_iter()
            .zip(splitted_keys)
            .map(|(df, keys)| {
                let gb = df.group_by_with_series(keys.into(), false, maintain_order)?;
                let groups = gb.get_groups();

                let mut columns = gb.keys();
                // don't naively call par_iter here, it will segfault in rayon
                // if you do, throw it on the POOL threadpool.
                let agg_columns = phys_aggs
                    .iter()
                    .map(|expr| {
                        let agg_expr = expr.as_partitioned_aggregator().unwrap();
                        let agg = agg_expr.evaluate_partitioned(&df, groups, state)?;
                        Ok(if agg.len() != groups.len() {
                            polars_ensure!(agg.len() == 1, agg_len = agg.len(), groups.len());
                            match groups.len() {
                                0 => agg.clear(),
                                len => agg.new_from_index(0, len),
                            }
                        } else {
                            agg
                        })
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;

                columns.extend_from_slice(&agg_columns);

                let df = DataFrame::new(columns)?;
                Ok((df, gb.keys()))
            })
            .collect()
    })
}

fn estimate_unique_count(keys: &[Series], mut sample_size: usize) -> PolarsResult<usize> {
    // https://stats.stackexchange.com/a/19090/147321
    // estimated unique size
    // u + ui / m (s - m)
    // s: set_size
    // m: sample_size
    // u: total unique groups counted in sample
    // ui: groups with single unique value counted in sample
    let set_size = keys[0].len();
    if set_size < sample_size {
        sample_size = set_size;
    }

    let finish = |groups: &GroupsProxy| {
        let u = groups.len() as f64;
        let ui = if groups.len() == sample_size {
            u
        } else {
            groups.iter().filter(|g| g.len() == 1).count() as f64
        };

        (u + (ui / sample_size as f64) * (set_size - sample_size) as f64) as usize
    };

    if keys.len() == 1 {
        // we sample as that will work also with sorted data.
        // not that sampling without replacement is very very expensive. don't do that.
        let s = keys[0].sample_n(sample_size, true, false, None).unwrap();
        // fast multi-threaded way to get unique.
        let groups = s.group_tuples(true, false)?;
        Ok(finish(&groups))
    } else {
        let offset = (keys[0].len() / 2) as i64;
        let keys = keys
            .iter()
            .map(|s| s.slice(offset, sample_size))
            .collect::<Vec<_>>();
        let df = unsafe { DataFrame::new_no_checks(keys) };
        let names = df.get_column_names();
        let gb = df.group_by(names).unwrap();
        Ok(finish(gb.get_groups()))
    }
}

// Lower this at debug builds so that we hit this in the test suite.
#[cfg(debug_assertions)]
const PARTITION_LIMIT: usize = 15;
#[cfg(not(debug_assertions))]
const PARTITION_LIMIT: usize = 1000;

// Checks if we should run normal or default aggregation
// by sampling data.
fn can_run_partitioned(
    keys: &[Series],
    original_df: &DataFrame,
    state: &ExecutionState,
    from_partitioned_ds: bool,
) -> PolarsResult<bool> {
    if !keys
        .iter()
        .take(1)
        .all(|s| matches!(s.is_sorted_flag(), IsSorted::Not))
    {
        if state.verbose() {
            eprintln!("FOUND SORTED KEY: running default HASH AGGREGATION")
        }
        Ok(false)
    } else if std::env::var("POLARS_NO_PARTITION").is_ok() {
        if state.verbose() {
            eprintln!("POLARS_NO_PARTITION set: running default HASH AGGREGATION")
        }
        Ok(false)
    } else if std::env::var("POLARS_FORCE_PARTITION").is_ok() {
        if state.verbose() {
            eprintln!("POLARS_FORCE_PARTITION set: running partitioned HASH AGGREGATION")
        }
        Ok(true)
    } else if original_df.height() < PARTITION_LIMIT && !cfg!(test) {
        if state.verbose() {
            eprintln!("DATAFRAME < {PARTITION_LIMIT} rows: running default HASH AGGREGATION")
        }
        Ok(false)
    } else {
        // below this boundary we assume the partitioned group_by will be faster
        let unique_count_boundary = std::env::var("POLARS_PARTITION_UNIQUE_COUNT")
            .map(|s| s.parse::<usize>().unwrap())
            .unwrap_or(1000);

        let (unique_estimate, sampled_method) = match (keys.len(), keys[0].dtype()) {
            #[cfg(feature = "dtype-categorical")]
            (1, DataType::Categorical(Some(rev_map), _) | DataType::Enum(Some(rev_map), _)) => {
                (rev_map.len(), "known")
            },
            _ => {
                // sqrt(N) is a good sample size as it remains low on large numbers
                // it is better than taking a fraction as it saturates
                let sample_size = (original_df.height() as f64).powf(0.5) as usize;

                // we never sample less than 100 data points.
                let sample_size = std::cmp::max(100, sample_size);
                (estimate_unique_count(keys, sample_size)?, "estimated")
            },
        };
        if state.verbose() {
            eprintln!("{sampled_method} unique values: {unique_estimate}");
        }

        if from_partitioned_ds {
            let estimated_cardinality = unique_estimate as f32 / original_df.height() as f32;
            if estimated_cardinality < 0.4 {
                if state.verbose() {
                    eprintln!("PARTITIONED DS");
                }
                Ok(true)
            } else {
                if state.verbose() {
                    eprintln!("PARTITIONED DS: estimated cardinality: {estimated_cardinality} exceeded the boundary: 0.4, running default HASH AGGREGATION");
                }
                Ok(false)
            }
        } else if unique_estimate > unique_count_boundary {
            if state.verbose() {
                eprintln!("estimated unique count: {unique_estimate} exceeded the boundary: {unique_count_boundary}, running default HASH AGGREGATION")
            }
            Ok(false)
        } else {
            Ok(true)
        }
    }
}

impl PartitionGroupByExec {
    fn execute_impl(
        &mut self,
        state: &mut ExecutionState,
        mut original_df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        let (splitted_dfs, splitted_keys) = {
            // already get the keys. This is the very last minute decision which group_by method we choose.
            // If the column is a categorical, we know the number of groups we have and can decide to continue
            // partitioned or go for the standard group_by. The partitioned is likely to be faster on a small number
            // of groups.
            let keys = self.keys(&original_df, state)?;

            if !can_run_partitioned(&keys, &original_df, state, self.from_partitioned_ds)? {
                return group_by_helper(
                    original_df,
                    keys,
                    &self.phys_aggs,
                    None,
                    state,
                    self.maintain_order,
                    self.slice,
                );
            }

            if state.verbose() {
                eprintln!("run PARTITIONED HASH AGGREGATION")
            }

            // Run the partitioned aggregations
            let n_threads = POOL.current_num_threads();

            run_partitions(
                &mut original_df,
                self,
                state,
                n_threads,
                self.maintain_order,
            )?
        };

        // MERGE phase

        let df = accumulate_dataframes_vertical(splitted_dfs)?;
        let keys = splitted_keys
            .into_iter()
            .reduce(|mut acc, e| {
                acc.iter_mut().zip(e).for_each(|(acc, e)| {
                    let _ = acc.append(&e);
                });
                acc
            })
            .unwrap();

        // the partitioned group_by has added columns so we must update the schema.
        state.set_schema(self.output_schema.clone());

        // merge and hash aggregate again

        // first get mutable access and optionally sort
        let gb = df.group_by_with_series(keys, true, self.maintain_order)?;
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
            let out: PolarsResult<Vec<_>> = self
                .phys_aggs
                .par_iter()
                // we slice the keys off and finalize every aggregation
                .zip(&df.get_columns()[self.phys_keys.len()..])
                .map(|(expr, partitioned_s)| {
                    let agg_expr = expr.as_partitioned_aggregator().unwrap();
                    agg_expr.finalize(partitioned_s.clone(), groups, state)
                })
                .collect();

            out
        };
        let (mut columns, agg_columns): (Vec<_>, _) = POOL.join(get_columns, get_agg);

        columns.extend(agg_columns?);
        state.clear_schema_cache();

        Ok(DataFrame::new(columns).unwrap())
    }
}

impl Executor for PartitionGroupByExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run PartitionGroupbyExec")
            }
        }
        let original_df = self.input.execute(state)?;

        let profile_name = if state.has_node_timer() {
            let by = self
                .phys_keys
                .iter()
                .map(|s| Ok(s.to_field(&self.input_schema)?.name))
                .collect::<PolarsResult<Vec<_>>>()?;
            let name = comma_delimited("group_by_partitioned".to_string(), &by);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };
        if state.has_node_timer() {
            let new_state = state.clone();
            new_state.record(|| self.execute_impl(state, original_df), profile_name)
        } else {
            self.execute_impl(state, original_df)
        }
    }
}
