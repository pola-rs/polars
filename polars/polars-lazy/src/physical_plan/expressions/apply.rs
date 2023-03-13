use std::borrow::Cow;
use std::sync::Arc;

use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use polars_core::POOL;
#[cfg(feature = "parquet")]
use polars_io::parquet::predicates::BatchStats;
#[cfg(feature = "parquet")]
use polars_io::predicates::StatsEvaluator;
#[cfg(feature = "parquet")]
use polars_plan::dsl::FunctionExpr;
use rayon::prelude::*;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct ApplyExpr {
    pub inputs: Vec<Arc<dyn PhysicalExpr>>,
    pub function: SpecialEq<Arc<dyn SeriesUdf>>,
    pub expr: Expr,
    pub collect_groups: ApplyOptions,
    pub auto_explode: bool,
    pub allow_rename: bool,
    pub pass_name_to_apply: bool,
    pub input_schema: Option<SchemaRef>,
}

impl ApplyExpr {
    pub(crate) fn new_minimal(
        inputs: Vec<Arc<dyn PhysicalExpr>>,
        function: SpecialEq<Arc<dyn SeriesUdf>>,
        expr: Expr,
        collect_groups: ApplyOptions,
    ) -> Self {
        Self {
            inputs,
            function,
            expr,
            collect_groups,
            auto_explode: false,
            allow_rename: false,
            pass_name_to_apply: false,
            input_schema: None,
        }
    }

    #[allow(clippy::ptr_arg)]
    fn prepare_multiple_inputs<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Vec<AggregationContext<'a>>> {
        POOL.install(|| {
            self.inputs
                .par_iter()
                .map(|e| e.evaluate_on_groups(df, groups, state))
                .collect()
        })
    }

    fn finish_apply_groups<'a>(
        &self,
        mut ac: AggregationContext<'a>,
        ca: ListChunked,
    ) -> PolarsResult<AggregationContext<'a>> {
        let all_unit_len = all_unit_length(&ca);
        if all_unit_len && self.auto_explode {
            ac.with_series(ca.explode().unwrap().into_series(), true, Some(&self.expr))?;
            ac.update_groups = UpdateGroups::No;
        } else {
            ac.with_series(ca.into_series(), true, Some(&self.expr))?;
            ac.with_update_groups(UpdateGroups::WithSeriesLen);
        }
        Ok(ac)
    }

    fn get_input_schema(&self, df: &DataFrame) -> Cow<Schema> {
        match &self.input_schema {
            Some(schema) => Cow::Borrowed(schema.as_ref()),
            None => Cow::Owned(df.schema()),
        }
    }

    /// evaluates and flattens `Option<Series>` to `Series`.
    fn eval_and_flatten(&self, inputs: &mut [Series]) -> PolarsResult<Series> {
        self.function.call_udf(inputs).map(|opt_out| {
            opt_out.unwrap_or_else(|| {
                let field = self.to_field(self.input_schema.as_ref().unwrap()).unwrap();
                Series::full_null(field.name(), 1, field.data_type())
            })
        })
    }
    fn apply_single_group_aware<'a>(
        &self,
        mut ac: AggregationContext<'a>,
    ) -> PolarsResult<AggregationContext<'a>> {
        let s = ac.series();

        polars_ensure!(
            !matches!(ac.agg_state(), AggState::AggregatedFlat(_)),
            expr = self.expr,
            ComputeError: "cannot aggregate, the column is already aggregated",
        );

        let name = s.name().to_string();
        let agg = ac.aggregated();
        // collection of empty list leads to a null dtype
        // see: #3687
        if agg.len() == 0 {
            // create input for the function to determine the output dtype
            // see #3946
            let agg = agg.list().unwrap();
            let input_dtype = agg.inner_dtype();

            let input = Series::full_null("", 0, &input_dtype);

            let output = self.eval_and_flatten(&mut [input])?;
            let ca = ListChunked::full(&name, &output, 0);
            return self.finish_apply_groups(ac, ca);
        }

        let mut ca: ListChunked = POOL.install(|| {
            agg.list()
                .unwrap()
                .par_iter()
                .map(|opt_s| match opt_s {
                    None => Ok(None),
                    Some(mut s) => {
                        if self.pass_name_to_apply {
                            s.rename(&name);
                        }
                        let mut container = [s];
                        self.function.call_udf(&mut container)
                    }
                })
                .collect::<PolarsResult<_>>()
        })?;

        ca.rename(&name);
        self.finish_apply_groups(ac, ca)
    }

    /// Apply elementwise e.g. ignore the group/list indices
    fn apply_single_elementwise<'a>(
        &self,
        mut ac: AggregationContext<'a>,
    ) -> PolarsResult<AggregationContext<'a>> {
        let (s, aggregated) = match ac.agg_state() {
            AggState::AggregatedList(s) => {
                let ca = s.list().unwrap();
                let out = ca.apply_to_inner(&|s| self.eval_and_flatten(&mut [s]))?;
                (out.into_series(), true)
            }
            AggState::AggregatedFlat(s) => (self.eval_and_flatten(&mut [s.clone()])?, true),
            AggState::NotAggregated(s) | AggState::Literal(s) => {
                (self.eval_and_flatten(&mut [s.clone()])?, false)
            }
        };

        ac.with_series(s, aggregated, Some(&self.expr))?;
        Ok(ac)
    }
    fn apply_multiple_group_aware<'a>(
        &self,
        mut acs: Vec<AggregationContext<'a>>,
        df: &DataFrame,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut container = vec![Default::default(); acs.len()];
        let schema = self.get_input_schema(df);
        let field = self.to_field(&schema)?;

        // aggregate representation of the aggregation contexts
        // then unpack the lists and finally create iterators from this list chunked arrays.
        let mut iters = acs
            .iter_mut()
            .map(|ac| ac.iter_groups())
            .collect::<Vec<_>>();

        // length of the items to iterate over
        let len = iters[0].size_hint().0;

        if len == 0 {
            let out = Series::full_null(field.name(), 0, &field.dtype);

            drop(iters);
            // take the first aggregation context that as that is the input series
            let mut ac = acs.swap_remove(0);
            ac.with_series(out, true, Some(&self.expr))?;
            return Ok(ac);
        }

        let mut ca: ListChunked = (0..len)
            .map(|_| {
                container.clear();
                for iter in &mut iters {
                    match iter.next().unwrap() {
                        None => return Ok(None),
                        Some(s) => container.push(s.deep_clone()),
                    }
                }
                self.function.call_udf(&mut container)
            })
            .collect::<PolarsResult<_>>()?;

        ca.rename(&field.name);
        drop(iters);

        // take the first aggregation context that as that is the input series
        let ac = acs.swap_remove(0);
        self.finish_apply_groups(ac, ca)
    }
}

fn all_unit_length(ca: &ListChunked) -> bool {
    assert_eq!(ca.chunks().len(), 1);
    let list_arr = ca.downcast_iter().next().unwrap();
    let offset = list_arr.offsets().as_slice();
    (offset[offset.len() - 1] as usize) == list_arr.len()
}

fn check_map_output_len(input_len: usize, output_len: usize, expr: &Expr) -> PolarsResult<()> {
    polars_ensure!(
        input_len == output_len, expr = expr, ComputeError:
        "output length of `map` must be equal to that of the input length; \
        consider using `apply` instead"
    );
    Ok(())
}

impl PhysicalExpr for ApplyExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let mut inputs = POOL.install(|| {
            self.inputs
                .par_iter()
                .map(|e| e.evaluate(df, state))
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        if self.allow_rename {
            return self.eval_and_flatten(&mut inputs);
        }
        let in_name = inputs[0].name().to_string();
        let mut out = self.eval_and_flatten(&mut inputs)?;
        if in_name != out.name() {
            out.rename(&in_name);
        }
        Ok(out)
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        if self.inputs.len() == 1 {
            let mut ac = self.inputs[0].evaluate_on_groups(df, groups, state)?;

            match self.collect_groups {
                ApplyOptions::ApplyList => {
                    let s = self.eval_and_flatten(&mut [ac.aggregated()])?;
                    ac.with_series(s, true, Some(&self.expr))?;
                    Ok(ac)
                }
                ApplyOptions::ApplyGroups => self.apply_single_group_aware(ac),
                ApplyOptions::ApplyFlat => self.apply_single_elementwise(ac),
            }
        } else {
            let mut acs = self.prepare_multiple_inputs(df, groups, state)?;

            match self.collect_groups {
                ApplyOptions::ApplyList => {
                    let mut s = acs.iter_mut().map(|ac| ac.aggregated()).collect::<Vec<_>>();
                    let s = self.eval_and_flatten(&mut s)?;
                    // take the first aggregation context that as that is the input series
                    let mut ac = acs.swap_remove(0);
                    ac.with_update_groups(UpdateGroups::WithGroupsLen);
                    ac.with_series(s, true, Some(&self.expr))?;
                    Ok(ac)
                }
                ApplyOptions::ApplyGroups => self.apply_multiple_group_aware(acs, df),
                ApplyOptions::ApplyFlat => {
                    if acs
                        .iter()
                        .any(|ac| matches!(ac.agg_state(), AggState::AggregatedList(_)))
                    {
                        self.apply_multiple_group_aware(acs, df)
                    } else {
                        apply_multiple_elementwise(acs, self.function.as_ref(), &self.expr)
                    }
                }
            }
        }
    }
    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.expr.to_field(input_schema, Context::Default)
    }
    fn is_valid_aggregation(&self) -> bool {
        matches!(self.collect_groups, ApplyOptions::ApplyGroups)
    }
    #[cfg(feature = "parquet")]
    fn as_stats_evaluator(&self) -> Option<&dyn polars_io::predicates::StatsEvaluator> {
        let function = match &self.expr {
            Expr::Function { function, .. } => function,
            _ => return None,
        };

        match function {
            FunctionExpr::IsNull => Some(self),
            #[cfg(feature = "is_in")]
            FunctionExpr::IsIn => Some(self),
            _ => None,
        }
    }
    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        if self.inputs.len() == 1 && matches!(self.collect_groups, ApplyOptions::ApplyFlat) {
            Some(self)
        } else {
            None
        }
    }
}

fn apply_multiple_elementwise<'a>(
    mut acs: Vec<AggregationContext<'a>>,
    function: &dyn SeriesUdf,
    expr: &Expr,
) -> PolarsResult<AggregationContext<'a>> {
    match acs.first().unwrap().agg_state() {
        // a fast path that doesn't drop groups of the first arg
        // this doesn't require group re-computation
        AggState::AggregatedList(s) => {
            let ca = s.list().unwrap();

            let other = acs[1..]
                .iter()
                .map(|ac| ac.flat_naive().into_owned())
                .collect::<Vec<_>>();

            let out = ca.apply_to_inner(&|s| {
                let mut args = vec![s];
                args.extend_from_slice(&other);
                let out = function.call_udf(&mut args)?.unwrap();
                Ok(out)
            })?;
            let mut ac = acs.swap_remove(0);
            ac.with_series(out.into_series(), true, None)?;
            Ok(ac)
        }
        _ => {
            let mut s = acs
                .iter_mut()
                .enumerate()
                .map(|(i, ac)| {
                    // make sure the groups are updated because we are about to throw away
                    // the series length information
                    // only on first iteration
                    if let (0, UpdateGroups::WithSeriesLen) = (i, &ac.update_groups) {
                        ac.groups();
                    }

                    ac.flat_naive().into_owned()
                })
                .collect::<Vec<_>>();

            let input_len = s[0].len();
            let s = function.call_udf(&mut s)?.unwrap();
            check_map_output_len(input_len, s.len(), expr)?;

            // take the first aggregation context that as that is the input series
            let mut ac = acs.swap_remove(0);
            ac.with_series(s, false, None)?;
            Ok(ac)
        }
    }
}

#[cfg(feature = "parquet")]
impl StatsEvaluator for ApplyExpr {
    fn should_read(&self, stats: &BatchStats) -> PolarsResult<bool> {
        let read = self.should_read_impl(stats)?;

        let state = ExecutionState::new();

        if state.verbose() && read {
            eprintln!("parquet file must be read, statistics not sufficient for predicate.")
        } else if state.verbose() && !read {
            eprintln!("parquet file can be skipped, the statistics were sufficient to apply the predicate.")
        };

        Ok(read)
    }
}

#[cfg(feature = "parquet")]
impl ApplyExpr {
    fn should_read_impl(&self, stats: &BatchStats) -> PolarsResult<bool> {
        let (function, input) = match &self.expr {
            Expr::Function {
                function, input, ..
            } => (function, input),
            _ => return Ok(true),
        };

        match function {
            FunctionExpr::IsNull => {
                let root = expr_to_leaf_column_name(&self.expr)?;

                match stats.get_stats(&root).ok() {
                    Some(st) => match st.null_count() {
                        Some(0) => Ok(false),
                        _ => Ok(true),
                    },
                    None => Ok(true),
                }
            }
            #[cfg(feature = "is_in")]
            FunctionExpr::IsIn => {
                let root = match expr_to_leaf_column_name(&input[0]) {
                    Ok(root) => root,
                    Err(_) => return Ok(true),
                };

                let input: &Series = match &input[1] {
                    Expr::Literal(LiteralValue::Series(s)) => s,
                    _ => return Ok(true),
                };

                match stats.get_stats(&root).ok() {
                    Some(st) => {
                        let min = match st.to_min() {
                            Some(min) => min,
                            None => return Ok(true),
                        };

                        let max = match st.to_max() {
                            Some(max) => max,
                            None => return Ok(true),
                        };

                        // all wanted values are smaller than minimum
                        // don't need to read
                        if ChunkCompare::<&Series>::lt(input, &min)
                            .ok()
                            .map(|ca| ca.all())
                            == Some(true)
                        {
                            return Ok(false);
                        }

                        // all wanted values are bigger than maximum
                        // don't need to read
                        if ChunkCompare::<&Series>::gt(input, &max)
                            .ok()
                            .map(|ca| ca.all())
                            == Some(true)
                        {
                            return Ok(false);
                        }

                        Ok(true)
                    }
                    None => Ok(true),
                }
            }
            _ => Ok(true),
        }
    }
}

impl PartitionedAggregation for ApplyExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        let a = self.inputs[0].as_partitioned_aggregator().unwrap();
        let s = a.evaluate_partitioned(df, groups, state)?;

        if self.allow_rename {
            return self.eval_and_flatten(&mut [s]);
        }
        let in_name = s.name().to_string();
        let mut out = self.eval_and_flatten(&mut [s])?;
        if in_name != out.name() {
            out.rename(&in_name);
        }
        Ok(out)
    }

    fn finalize(
        &self,
        partitioned: Series,
        _groups: &GroupsProxy,
        _state: &ExecutionState,
    ) -> PolarsResult<Series> {
        Ok(partitioned)
    }
}
