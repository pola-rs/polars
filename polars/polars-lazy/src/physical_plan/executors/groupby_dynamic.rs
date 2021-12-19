use super::*;
use crate::prelude::utils::as_aggregated;
use polars_core::frame::groupby::DynamicGroupOptions;
use polars_core::POOL;
use rayon::prelude::*;

pub(crate) struct GroupByDynamicExec {
    pub(crate) input: Box<dyn Executor>,
    // we will use this later
    #[allow(dead_code)]
    pub(crate) keys: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) aggs: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) options: DynamicGroupOptions,
}

impl Executor for GroupByDynamicExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        #[cfg(feature = "dynamic_groupby")]
        {
            let df = self.input.execute(state)?;
            let keys = self
                .keys
                .iter()
                .map(|e| e.evaluate(&df, state))
                .collect::<Result<Vec<_>>>()?;

            let (time_key, keys, groups) = df.groupby_dynamic(keys, &self.options)?;

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

            let mut columns = Vec::with_capacity(agg_columns.len() + 1 + keys.len());
            columns.extend(keys);
            columns.push(time_key);
            columns.extend(agg_columns.into_iter().flatten());

            DataFrame::new(columns)
        }
        #[cfg(not(feature = "dynamic_groupby"))]
        panic!("activate feature dynamic_groupby")
    }
}
