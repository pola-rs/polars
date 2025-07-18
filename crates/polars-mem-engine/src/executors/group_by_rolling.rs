use polars_utils::unique_column_name;

use super::*;

#[cfg_attr(not(feature = "dynamic_group_by"), allow(dead_code))]
pub(crate) struct GroupByRollingExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) keys: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) aggs: Vec<Arc<dyn PhysicalExpr>>,
    #[cfg(feature = "dynamic_group_by")]
    pub(crate) options: RollingGroupOptions,
    pub(crate) input_schema: SchemaRef,
    pub(crate) slice: Option<(i64, usize)>,
    pub(crate) apply: Option<Arc<dyn DataFrameUdf>>,
}

pub(super) fn sort_and_groups(
    df: &mut DataFrame,
    keys: &mut Vec<Column>,
) -> PolarsResult<Vec<[IdxSize; 2]>> {
    let encoded = row_encode::encode_rows_vertical_par_unordered(keys)?;
    let encoded = encoded.rechunk().into_owned();
    let encoded = encoded.with_name(unique_column_name());
    let idx = encoded.arg_sort(SortOptions {
        maintain_order: true,
        ..Default::default()
    });

    let encoded = unsafe {
        df.with_column_unchecked(encoded.into_series().into());

        // If not sorted on keys, sort.
        let idx_s = idx.clone().into_series();
        if !idx_s.is_sorted(Default::default()).unwrap() {
            let (df_ordered, keys_ordered) = POOL.join(
                || df.take_unchecked(&idx),
                || {
                    keys.iter()
                        .map(|c| c.take_unchecked(&idx))
                        .collect::<Vec<_>>()
                },
            );
            *df = df_ordered;
            *keys = keys_ordered;
        }

        df.get_columns_mut().pop().unwrap()
    };
    let encoded = encoded.as_materialized_series();
    let encoded = encoded.binary_offset().unwrap();
    let encoded = encoded.with_sorted_flag(polars_core::series::IsSorted::Ascending);
    let groups = encoded.group_tuples(true, false).unwrap();

    let GroupsType::Slice { groups, .. } = groups else {
        // memory would explode
        unreachable!();
    };
    Ok(groups)
}

impl GroupByRollingExec {
    #[cfg(feature = "dynamic_group_by")]
    fn execute_impl(
        &mut self,
        state: &ExecutionState,
        mut df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        df.as_single_chunk_par();

        let mut keys = self
            .keys
            .iter()
            .map(|e| e.evaluate(&df, state))
            .collect::<PolarsResult<Vec<_>>>()?;

        let group_by = if !self.keys.is_empty() {
            Some(sort_and_groups(&mut df, &mut keys)?)
        } else {
            None
        };

        let (mut time_key, groups) = df.rolling(group_by, &self.options)?;

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
            groups = sliced_groups.as_ref().unwrap();

            time_key = time_key.slice(offset, len);
            for k in &mut keys {
                *k = k.slice(offset, len);
            }
        }

        let agg_columns = evaluate_aggs(&df, &self.aggs, groups, state)?;

        let mut columns = Vec::with_capacity(agg_columns.len() + 1 + keys.len());
        columns.extend_from_slice(&keys);
        columns.push(time_key);
        columns.extend(agg_columns);

        DataFrame::new(columns)
    }
}

impl Executor for GroupByRollingExec {
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
                eprintln!("run GroupbyRollingExec")
            }
        }
        let df = self.input.execute(state)?;
        let profile_name = if state.has_node_timer() {
            let by = self
                .keys
                .iter()
                .map(|s| Ok(s.to_field(&self.input_schema)?.name))
                .collect::<PolarsResult<Vec<_>>>()?;
            let name = comma_delimited("group_by_rolling".to_string(), &by);
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
