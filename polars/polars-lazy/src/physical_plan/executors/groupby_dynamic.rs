use super::*;

#[cfg_attr(not(feature = "dynamic_groupby"), allow(dead_code))]
pub(crate) struct GroupByDynamicExec {
    pub(crate) input: Box<dyn Executor>,
    // we will use this later
    #[allow(dead_code)]
    pub(crate) keys: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) aggs: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) options: DynamicGroupOptions,
    pub(crate) input_schema: SchemaRef,
}

impl Executor for GroupByDynamicExec {
    #[cfg(not(feature = "dynamic_groupby"))]
    fn execute(&mut self, _state: &ExecutionState) -> Result<DataFrame> {
        panic!("activate feature dynamic_groupby")
    }

    #[cfg(feature = "dynamic_groupby")]
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        use crate::prelude::{utils::as_aggregated, *};

        let df = self.input.execute(state)?;
        state.set_schema(self.input_schema.clone());
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

        state.clear_schema_cache();
        let mut columns = Vec::with_capacity(agg_columns.len() + 1 + keys.len());
        columns.extend_from_slice(&keys);
        columns.push(time_key);
        columns.extend(agg_columns.into_iter().flatten());

        DataFrame::new(columns)
    }
}
