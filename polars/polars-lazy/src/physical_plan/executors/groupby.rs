use super::*;
use crate::logical_plan::Context;
use crate::utils::rename_aexpr_root_name;
use polars_core::frame::groupby::partition::group_maps_to_group_index;
use polars_core::utils::{accumulate_dataframes_vertical, num_cpus, split_df};
use polars_core::POOL;
use rayon::prelude::*;

/// Take an input Executor and a multiple expressions
pub struct GroupByExec {
    input: Box<dyn Executor>,
    keys: Vec<Arc<dyn PhysicalExpr>>,
    aggs: Vec<Arc<dyn PhysicalExpr>>,
    apply: Option<Arc<dyn DataFrameUdf>>,
}

impl GroupByExec {
    pub(crate) fn new(
        input: Box<dyn Executor>,
        keys: Vec<Arc<dyn PhysicalExpr>>,
        aggs: Vec<Arc<dyn PhysicalExpr>>,
        apply: Option<Arc<dyn DataFrameUdf>>,
    ) -> Self {
        Self {
            input,
            keys,
            aggs,
            apply,
        }
    }
}

fn groupby_helper(
    df: DataFrame,
    keys: Vec<Series>,
    aggs: &[Arc<dyn PhysicalExpr>],
    apply: Option<&Arc<dyn DataFrameUdf>>,
    state: &ExecutionState,
) -> Result<DataFrame> {
    let gb = df.groupby_with_series(keys, true)?;
    if let Some(f) = apply {
        return gb.apply(|df| f.call_udf(df));
    }

    let groups = gb.get_groups();

    let mut columns = gb.keys();

    let agg_columns = POOL.install(|| {
        aggs
            .par_iter()
            .map(|expr| {
                let agg_expr = expr.as_agg_expr()?;
                let opt_agg = agg_expr.aggregate(&df, groups, state)?;
                if let Some(agg) = &opt_agg {
                    if agg.len() != groups.len() {
                        panic!(
                            "returned aggregation is a different length: {} than the group lengths: {}",
                            agg.len(),
                            groups.len()
                        )
                    }
                };
                Ok(opt_agg)
            })
            .collect::<Result<Vec<_>>>()
    })?;

    columns.extend(agg_columns.into_iter().flatten());

    let df = DataFrame::new_no_checks(columns);
    Ok(df)
}

impl Executor for GroupByExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let df = self.input.execute(state)?;
        let keys = self
            .keys
            .iter()
            .map(|e| e.evaluate(&df, state))
            .collect::<Result<_>>()?;
        groupby_helper(df, keys, &self.aggs, self.apply.as_ref(), state)
    }
}

/// Take an input Executor and a multiple expressions
pub struct PartitionGroupByExec {
    input: Box<dyn Executor>,
    keys: Vec<Arc<dyn PhysicalExpr>>,
    phys_aggs: Vec<Arc<dyn PhysicalExpr>>,
    aggs: Vec<Expr>,
}

impl PartitionGroupByExec {
    pub(crate) fn new(
        input: Box<dyn Executor>,
        keys: Vec<Arc<dyn PhysicalExpr>>,
        phys_aggs: Vec<Arc<dyn PhysicalExpr>>,
        aggs: Vec<Expr>,
    ) -> Self {
        Self {
            input,
            keys,
            phys_aggs,
            aggs,
        }
    }
}

impl Executor for PartitionGroupByExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let original_df = self.input.execute(state)?;

        // already get the keys. This is the very last minute decision which groupby method we choose.
        // If the column is a categorical, we know the number of groups we have and can decide to continue
        // partitioned or go for the standard groupby. The partitioned is likely to be faster on a small number
        // of groups.
        let keys = self
            .keys
            .iter()
            .map(|e| e.evaluate(&original_df, state))
            .collect::<Result<Vec<_>>>()?;

        // We only do partitioned groupby's on single keys aggregation.
        // This design choice seems ok, as cardinality rapidly increases with multiple columns
        debug_assert_eq!(keys.len(), 1);
        let key = &keys[0];
        if let Ok(ca) = key.categorical() {
            let cat_map = ca
                .get_categorical_map()
                .expect("categorical type has categorical_map");
            let frac = cat_map.len() as f32 / ca.len() as f32;
            // TODO! proper benchmark which boundary should be chosen.
            if frac > 0.3 {
                return groupby_helper(original_df, keys, &self.phys_aggs, None, state);
            }
        }
        if std::env::var("POLARS_NEW_PARTITION").is_ok() && !matches!(key.dtype(), DataType::Utf8) {
            dbg!("RUN PARTITIONED");
            let mut exec = PartitionGroupByExec2 {
                input: original_df,
                key: self.keys[0].clone(),
                phys_aggs: std::mem::take(&mut self.phys_aggs),
            };
            return exec.execute(state);
        }

        let mut expr_arena = Arena::with_capacity(64);

        // This will be the aggregation on the partition results. Due to the groupby
        // operation the column names have changed. This makes sure we can select the columns with
        // the new names. We also keep a hold on the names to make sure that we don't get a double
        // new name due to the double aggregation. These output_names will be used to rename the final
        // output
        let schema = original_df.schema();
        let aggs_and_names = self
            .aggs
            .iter()
            .map(|e| {
                let out_field = e.to_field(&schema, Context::Aggregation)?;
                let out_name = Arc::new(out_field.name().clone());
                let node = to_aexpr(e.clone(), &mut expr_arena);
                rename_aexpr_root_name(node, &mut expr_arena, out_name.clone())?;
                Ok((node, out_name))
            })
            .collect::<Result<Vec<_>>>()?;

        let planner = DefaultPlanner {};
        let outer_phys_aggs = aggs_and_names
            .iter()
            .map(|(e, _)| planner.create_physical_expr(*e, Context::Aggregation, &mut expr_arena))
            .collect::<Result<Vec<_>>>()?;

        let n_threads = num_cpus::get();
        // We do a partitioned groupby. Meaning that we first do the groupby operation arbitrarily
        // splitted on several threads. Than the final result we apply the same groupby again.
        let dfs = split_df(&original_df, n_threads)?;

        let dfs = POOL.install(|| {
            dfs.into_par_iter()
                .map(|df| {
                    let keys = self
                        .keys
                        .iter()
                        .map(|e| e.evaluate(&df, state))
                        .collect::<Result<Vec<_>>>()?;
                    let phys_aggs = &self.phys_aggs;
                    let gb = df.groupby_with_series(keys, false)?;
                    let groups = gb.get_groups();

                    let mut columns = gb.keys();
                    let agg_columns = phys_aggs
                        .iter()
                        .map(|expr| {
                            let agg_expr = expr.as_agg_expr()?;
                            let opt_agg = agg_expr.evaluate_partitioned(&df, groups, state)?;
                            if let Some(agg) = &opt_agg {
                                if agg[0].len() != groups.len() {
                                    panic!(
                                        "returned aggregation is a different length: {} than the group lengths: {}",
                                        agg.len(),
                                        groups.len()
                                    )
                                }
                            };
                            Ok(opt_agg)
                        }).collect::<Result<Vec<_>>>()?;

                    for agg in agg_columns.into_iter().flatten() {
                        for agg in agg {
                            columns.push(agg)
                        }
                    }

                    let df = DataFrame::new_no_checks(columns);
                    Ok(df)
                })
        }).collect::<Result<Vec<_>>>()?;

        let df = accumulate_dataframes_vertical(dfs)?;

        let keys = self
            .keys
            .iter()
            .map(|e| e.evaluate(&df, state))
            .collect::<Result<Vec<_>>>()?;

        // do the same on the outer results
        let gb = df.groupby_with_series(keys, true)?;
        let groups = gb.get_groups();

        let mut columns = gb.keys();
        let agg_columns = outer_phys_aggs
            .iter()
            .zip(aggs_and_names.iter().map(|(_, name)| name))
            .filter_map(|(expr, name)| {
                let agg_expr = expr.as_agg_expr().unwrap();
                // If None the column doesn't exist anymore.
                // For instance when summing a string this column will not be in the aggregation result
                let opt_agg = agg_expr.evaluate_partitioned_final(&df, groups, state).ok();
                opt_agg.map(|opt_s| {
                    opt_s.map(|mut s| {
                        s.rename(name);
                        s
                    })
                })
            });

        columns.extend(agg_columns.flatten());

        let df = DataFrame::new_no_checks(columns);
        Ok(df)
    }
}

/// Take an input Executor and a multiple expressions
pub struct PartitionGroupByExec2 {
    input: DataFrame,
    key: Arc<dyn PhysicalExpr>,
    phys_aggs: Vec<Arc<dyn PhysicalExpr>>,
}

impl Executor for PartitionGroupByExec2 {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let df = &self.input;
        let key = self.key.evaluate(df, state)?;
        let g_maps = key.group_maps();
        let key = {
            let key_idx = group_maps_to_group_index(&g_maps);
            // Safety:
            // Indexes of groups are in bounds
            unsafe { key.take_unchecked(&key_idx)? }
        };

        let agg_columns = self
            .phys_aggs
            .iter()
            .map(|expr| {
                let agg_expr = expr.as_agg_expr()?;
                agg_expr.evaluate_partitioned_2(&df, &g_maps, state)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut columns = Vec::with_capacity(agg_columns.len() + 1);
        columns.push(key);
        columns.extend(agg_columns.into_iter().flatten());

        Ok(DataFrame::new_no_checks(columns))
    }
}
