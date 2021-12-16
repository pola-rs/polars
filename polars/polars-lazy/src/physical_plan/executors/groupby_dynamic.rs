use super::*;
use crate::logical_plan::Context;
use crate::prelude::utils::as_aggregated;
use crate::utils::rename_aexpr_root_name;
use polars_core::utils::{accumulate_dataframes_vertical, split_df};
use polars_core::POOL;
use rayon::prelude::*;
use polars_core::frame::groupby::DynamicGroupOptions;

pub(crate) struct GroupByDynamicExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) keys: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) aggs: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) options: DynamicGroupOptions
}

impl Executor for GroupByDynamicExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        #[cfg(feature = "dynamic_groupby")]
        {
            let df = self.input.execute(state)?;

            // don't truncate all values
            // we truncate when we have collected the keys
            let mut options = self.options.clone();;
            options.truncate = false;
            let (key, groups) = df.groupby_dynamic(&options)?;

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

            let mut columns = Vec::with_capacity(agg_columns.len() + 1);
            columns.push(key);
            columns.extend(agg_columns.into_iter().flatten());

            dbg!(&columns);

            DataFrame::new(columns)
        }
        #[cfg(not(feature = "dynamic_groupby"))]
        panic!("activate feature dynamic_groupby")
    }
}
