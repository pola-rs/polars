#[cfg(feature = "dynamic_groupby")]
use polars_time::RollingGroupOptions;

use super::*;

#[cfg_attr(not(feature = "dynamic_groupby"), allow(dead_code))]
pub(crate) struct GroupByRollingExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) keys: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) aggs: Vec<Arc<dyn PhysicalExpr>>,
    #[cfg(feature = "dynamic_groupby")]
    pub(crate) options: RollingGroupOptions,
    pub(crate) input_schema: SchemaRef,
    pub(crate) slice: Option<(i64, usize)>,
}

impl GroupByRollingExec {
    #[cfg(feature = "dynamic_groupby")]
    fn execute_impl(
        &mut self,
        state: &mut ExecutionState,
        mut df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        df.as_single_chunk_par();

        let keys = self
            .keys
            .iter()
            .map(|e| e.evaluate(&df, state))
            .collect::<PolarsResult<Vec<_>>>()?;

        let (mut time_key, mut keys, groups) = df.groupby_rolling(keys, &self.options)?;

        let mut groups = &groups;
        #[allow(unused_assignments)]
        // it is unused because we only use it to keep the lifetime of sliced_group valid
        let mut sliced_groups = None;

        if let Some((offset, len)) = self.slice {
            sliced_groups = Some(groups.slice(offset, len));
            groups = sliced_groups.as_deref().unwrap();

            time_key = time_key.slice(offset, len);
        }

        // the ordering has changed due to the groupby
        if !keys.is_empty() {
            unsafe {
                match groups {
                    GroupsProxy::Idx(groups) => {
                        let first = groups.first();
                        // we don't use agg_first here, because the group
                        // can be empty, but we still want to know the first value
                        // of that group
                        for key in keys.iter_mut() {
                            *key = key.take_unchecked_from_slice(first).unwrap();
                        }
                    }
                    GroupsProxy::Slice { groups, .. } => {
                        for key in keys.iter_mut() {
                            let iter = &mut groups.iter().map(|[first, _len]| *first as usize)
                                as &mut dyn TakeIterator<Item = usize>;
                            *key = key.take_iter_unchecked(iter);
                        }
                    }
                }
            }
        };

        // a rolling groupby has overlapping windows
        state.set_has_overlapping_groups();

        let agg_columns = POOL.install(|| {
            self.aggs
                .par_iter()
                .map(|expr| {
                    let agg = expr.evaluate_on_groups(&df, groups, state)?.aggregated();
                    if agg.len() != groups.len() {
                        return Err(PolarsError::ComputeError(
                            format!("returned aggregation is a different length: {} than the group lengths: {}",
                                    agg.len(),
                                    groups.len()).into()
                        ))
                    }
                    Ok(agg)
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        let mut columns = Vec::with_capacity(agg_columns.len() + 1 + keys.len());
        columns.extend_from_slice(&keys);
        columns.push(time_key);
        columns.extend_from_slice(&agg_columns);

        DataFrame::new(columns)
    }
}

impl Executor for GroupByRollingExec {
    #[cfg(not(feature = "dynamic_groupby"))]
    fn execute(&mut self, _state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        panic!("activate feature dynamic_groupby")
    }

    #[cfg(feature = "dynamic_groupby")]
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run GroupbyRollingExec")
            }
        }
        let df = self.input.execute(state)?;
        let profile_name = if state.has_node_timer() {
            let by = self
                .keys
                .iter()
                .map(|s| Ok(s.to_field(&self.input_schema)?.name))
                .collect::<PolarsResult<Vec<_>>>()?;
            let name = column_delimited("groupby_rolling".to_string(), &by);
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
