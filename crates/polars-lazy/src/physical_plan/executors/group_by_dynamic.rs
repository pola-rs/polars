use super::*;

#[cfg_attr(not(feature = "dynamic_group_by"), allow(dead_code))]
pub(crate) struct GroupByDynamicExec {
    pub(crate) input: Box<dyn Executor>,
    // we will use this later
    #[allow(dead_code)]
    pub(crate) keys: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) aggs: Vec<Arc<dyn PhysicalExpr>>,
    #[cfg(feature = "dynamic_group_by")]
    pub(crate) options: DynamicGroupOptions,
    pub(crate) input_schema: SchemaRef,
    pub(crate) slice: Option<(i64, usize)>,
    pub(crate) apply: Option<Arc<dyn DataFrameUdf>>,
}

impl GroupByDynamicExec {
    #[cfg(feature = "dynamic_group_by")]
    fn execute_impl(
        &mut self,
        state: &ExecutionState,
        mut df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        df.as_single_chunk_par();
        let keys = self
            .keys
            .iter()
            .map(|e| e.evaluate(&df, state))
            .collect::<PolarsResult<Vec<_>>>()?;

        let (mut time_key, mut keys, groups) = df.group_by_dynamic(keys, &self.options)?;

        if let Some(f) = &self.apply {
            let gb = GroupBy::new(&df, vec![], groups, None);
            let out = gb.apply(move |df| f.call_udf(df))?;
            return Ok(if let Some((offset, len)) = self.slice {
                out.slice(offset, len)
            } else {
                out
            });
        }

        let mut groups = &groups;
        #[allow(unused_assignments)]
        // it is unused because we only use it to keep the lifetime of sliced_group valid
        let mut sliced_groups = None;

        if let Some((offset, len)) = self.slice {
            sliced_groups = Some(groups.slice(offset, len));
            groups = sliced_groups.as_deref().unwrap();

            time_key = time_key.slice(offset, len);

            // todo! optimize this, we can prevent an agg_first aggregation upstream
            // the ordering has changed due to the group_by
            for key in keys.iter_mut() {
                *key = key.slice(offset, len)
            }
        }

        let agg_columns = evaluate_aggs(&df, &self.aggs, groups, state)?;

        let mut columns = Vec::with_capacity(agg_columns.len() + 1 + keys.len());
        columns.extend_from_slice(&keys);
        columns.push(time_key);
        columns.extend_from_slice(&agg_columns);

        DataFrame::new(columns)
    }
}

impl Executor for GroupByDynamicExec {
    #[cfg(not(feature = "dynamic_group_by"))]
    fn execute(&mut self, _state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        panic!("activate feature dynamic_group_by")
    }

    #[cfg(feature = "dynamic_group_by")]
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run GroupbyDynamicExec")
            }
        }
        let df = self.input.execute(state)?;

        let profile_name = if state.has_node_timer() {
            let by = self
                .keys
                .iter()
                .map(|s| Ok(s.to_field(&self.input_schema)?.name))
                .collect::<PolarsResult<Vec<_>>>()?;
            let name = comma_delimited("group_by_dynamic".to_string(), &by);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        if state.has_node_timer() {
            let new_state = state.clone();
            new_state.record(|| self.execute_impl(state, df), profile_name)
        } else {
            self.execute_impl(state, df)
        }
    }
}
