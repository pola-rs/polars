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
    pub(crate) slice: Option<(i64, usize)>,
}

impl Executor for GroupByDynamicExec {
    #[cfg(not(feature = "dynamic_groupby"))]
    fn execute(&mut self, _state: &mut ExecutionState) -> Result<DataFrame> {
        panic!("activate feature dynamic_groupby")
    }

    #[cfg(feature = "dynamic_groupby")]
    fn execute(&mut self, state: &mut ExecutionState) -> Result<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run GroupbyDynamicExec")
            }
        }
        let mut df = self.input.execute(state)?;
        df.as_single_chunk_par();
        state.set_schema(self.input_schema.clone());
        let keys = self
            .keys
            .iter()
            .map(|e| e.evaluate(&df, state))
            .collect::<Result<Vec<_>>>()?;

        let (time_key, keys, groups) = df.groupby_dynamic(keys, &self.options)?;

        let mut groups = &groups;
        #[allow(unused_assignments)]
        // it is unused because we only use it to keep the lifetime of sliced_group valid
        let mut sliced_groups = None;

        if let Some((offset, len)) = self.slice {
            sliced_groups = Some(groups.slice(offset, len));
            groups = sliced_groups.as_deref().unwrap();
        }

        let agg_columns = POOL.install(|| {
                self.aggs
                    .par_iter()
                    .map(|expr| {
                        let agg = expr.evaluate_on_groups(&df, groups, state)?.finalize();
                        if agg.len() != groups.len() {
                            return Err(PolarsError::ComputeError(
                                format!("returned aggregation is a different length: {} than the group lengths: {}",
                                        agg.len(),
                                        groups.len()).into()
                            ))
                        }
                        Ok(agg)
                    })
                    .collect::<Result<Vec<_>>>()
            })?;

        state.clear_schema_cache();
        let mut columns = Vec::with_capacity(agg_columns.len() + 1 + keys.len());
        columns.extend_from_slice(&keys);
        columns.push(time_key);
        columns.extend_from_slice(&agg_columns);

        DataFrame::new(columns)
    }
}
