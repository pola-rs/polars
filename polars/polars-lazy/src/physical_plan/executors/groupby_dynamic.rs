use super::*;
use crate::logical_plan::Context;
use crate::prelude::utils::as_aggregated;
use crate::utils::rename_aexpr_root_name;
use polars_core::utils::{accumulate_dataframes_vertical, split_df};
use polars_core::POOL;
use rayon::prelude::*;
use polars_core::frame::groupby::DynamicGroupOptions;

pub struct GroupByDynamicExec {
    input: Box<dyn Executor>,
    keys: Vec<Arc<dyn PhysicalExpr>>,
    aggs: Vec<Arc<dyn PhysicalExpr>>,
    options: DynamicGroupOptions
}

impl Executor for GroupByDynamicExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let df = self.input.execute(state)?;

        let (df, groups) = df.groupby_dynamic(&self.options)?;
        let key = df.column(&self.options.time_column)?.clone();

        let agg_columns = POOL.install(|| {
            self.aggs
                .par_iter()
                .map(|expr| {
                    let opt_agg = as_aggregated(expr.as_ref(), &df, &groups, state)?;
                    if let Some(agg) = &opt_agg {
                        if agg.len() != groups.len() {
                            return Err(PolarsError::ComputeError(
                                format!("returned aggregation is a different length: {} than the group lengths: {}",
                                        agg.len(),
                                        groups.len()).into()
                            ))
                        }
                    };
                    Ok(opt_agg)
                })
                .collect::<Result<Vec<_>>>()

        })?;

        let mut columns= Vec::with_capacity(agg_columns.len() + 1);
        columns.push(key);
        columns.extend(agg_columns.into_iter().flatten());

        DataFrame::new(columns)
    }
}
