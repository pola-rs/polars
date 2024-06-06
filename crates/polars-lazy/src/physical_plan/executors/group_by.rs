use rayon::prelude::*;

use super::*;

pub(super) fn evaluate_aggs(
    df: &DataFrame,
    aggs: &[Arc<dyn PhysicalExpr>],
    groups: &GroupsProxy,
    state: &ExecutionState,
) -> PolarsResult<Vec<Series>> {
    POOL.install(|| {
        aggs.par_iter()
            .map(|expr| {
                let agg = expr.evaluate_on_groups(df, groups, state)?.finalize();
                polars_ensure!(agg.len() == groups.len(), agg_len = agg.len(), groups.len());
                Ok(agg)
            })
            .collect::<PolarsResult<Vec<_>>>()
    })
}

/// Take an input Executor and a multiple expressions
pub struct GroupByExec {
    input: Box<dyn Executor>,
    keys: Vec<Arc<dyn PhysicalExpr>>,
    aggs: Vec<Arc<dyn PhysicalExpr>>,
    apply: Option<Arc<dyn DataFrameUdf>>,
    maintain_order: bool,
    input_schema: SchemaRef,
    slice: Option<(i64, usize)>,
}

impl GroupByExec {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input: Box<dyn Executor>,
        keys: Vec<Arc<dyn PhysicalExpr>>,
        aggs: Vec<Arc<dyn PhysicalExpr>>,
        apply: Option<Arc<dyn DataFrameUdf>>,
        maintain_order: bool,
        input_schema: SchemaRef,
        slice: Option<(i64, usize)>,
    ) -> Self {
        Self {
            input,
            keys,
            aggs,
            apply,
            maintain_order,
            input_schema,
            slice,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn group_by_helper(
    mut df: DataFrame,
    keys: Vec<Series>,
    aggs: &[Arc<dyn PhysicalExpr>],
    apply: Option<Arc<dyn DataFrameUdf>>,
    state: &ExecutionState,
    maintain_order: bool,
    slice: Option<(i64, usize)>,
) -> PolarsResult<DataFrame> {
    df.as_single_chunk_par();
    let gb = df.group_by_with_series(keys, true, maintain_order)?;

    if let Some(f) = apply {
        return gb.apply(move |df| f.call_udf(df));
    }

    let mut groups = gb.get_groups();

    #[allow(unused_assignments)]
    // it is unused because we only use it to keep the lifetime of sliced_group valid
    let mut sliced_groups = None;

    if let Some((offset, len)) = slice {
        sliced_groups = Some(groups.slice(offset, len));
        groups = sliced_groups.as_deref().unwrap();
    }

    let (mut columns, agg_columns) = POOL.install(|| {
        let get_columns = || gb.keys_sliced(slice);

        let get_agg = || evaluate_aggs(&df, aggs, groups, state);

        rayon::join(get_columns, get_agg)
    });
    let agg_columns = agg_columns?;

    columns.extend_from_slice(&agg_columns);
    DataFrame::new(columns)
}

impl GroupByExec {
    fn execute_impl(&mut self, state: &ExecutionState, df: DataFrame) -> PolarsResult<DataFrame> {
        let keys = self
            .keys
            .iter()
            .map(|e| e.evaluate(&df, state))
            .collect::<PolarsResult<_>>()?;
        group_by_helper(
            df,
            keys,
            &self.aggs,
            self.apply.take(),
            state,
            self.maintain_order,
            self.slice,
        )
    }
}

impl Executor for GroupByExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run GroupbyExec")
            }
        }
        if state.verbose() {
            eprintln!("keys/aggregates are not partitionable: running default HASH AGGREGATION")
        }
        let df = self.input.execute(state)?;

        let profile_name = if state.has_node_timer() {
            let by = self
                .keys
                .iter()
                .map(|s| Ok(s.to_field(&self.input_schema)?.name))
                .collect::<PolarsResult<Vec<_>>>()?;
            let name = comma_delimited("group_by".to_string(), &by);
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
