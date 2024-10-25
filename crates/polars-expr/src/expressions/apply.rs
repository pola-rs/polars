use std::borrow::Cow;

use polars_core::chunked_array::builder::get_list_builder;
use polars_core::prelude::*;
use polars_core::POOL;
#[cfg(feature = "parquet")]
use polars_io::predicates::{BatchStats, StatsEvaluator};
#[cfg(feature = "is_between")]
use polars_ops::prelude::ClosedInterval;
use rayon::prelude::*;

use super::*;
use crate::expressions::{
    AggState, AggregationContext, PartitionedAggregation, PhysicalExpr, UpdateGroups,
};

pub struct ApplyExpr {
    inputs: Vec<Arc<dyn PhysicalExpr>>,
    function: SpecialEq<Arc<dyn ColumnsUdf>>,
    expr: Expr,
    collect_groups: ApplyOptions,
    function_returns_scalar: bool,
    function_operates_on_scalar: bool,
    allow_rename: bool,
    pass_name_to_apply: bool,
    input_schema: SchemaRef,
    allow_threading: bool,
    check_lengths: bool,
    allow_group_aware: bool,
    output_field: Field,
}

impl ApplyExpr {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        inputs: Vec<Arc<dyn PhysicalExpr>>,
        function: SpecialEq<Arc<dyn ColumnsUdf>>,
        expr: Expr,
        options: FunctionOptions,
        allow_threading: bool,
        input_schema: SchemaRef,
        output_field: Field,
        returns_scalar: bool,
    ) -> Self {
        #[cfg(debug_assertions)]
        if matches!(options.collect_groups, ApplyOptions::ElementWise)
            && options.flags.contains(FunctionFlags::RETURNS_SCALAR)
        {
            panic!("expr {:?} is not implemented correctly. 'returns_scalar' and 'elementwise' are mutually exclusive", expr)
        }

        Self {
            inputs,
            function,
            expr,
            collect_groups: options.collect_groups,
            function_returns_scalar: options.flags.contains(FunctionFlags::RETURNS_SCALAR),
            function_operates_on_scalar: returns_scalar,
            allow_rename: options.flags.contains(FunctionFlags::ALLOW_RENAME),
            pass_name_to_apply: options.flags.contains(FunctionFlags::PASS_NAME_TO_APPLY),
            input_schema,
            allow_threading,
            check_lengths: options.check_lengths(),
            allow_group_aware: options.flags.contains(FunctionFlags::ALLOW_GROUP_AWARE),
            output_field,
        }
    }

    #[allow(clippy::ptr_arg)]
    fn prepare_multiple_inputs<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Vec<AggregationContext<'a>>> {
        let f = |e: &Arc<dyn PhysicalExpr>| e.evaluate_on_groups(df, groups, state);
        if self.allow_threading {
            POOL.install(|| self.inputs.par_iter().map(f).collect())
        } else {
            self.inputs.iter().map(f).collect()
        }
    }

    fn finish_apply_groups<'a>(
        &self,
        mut ac: AggregationContext<'a>,
        ca: ListChunked,
    ) -> PolarsResult<AggregationContext<'a>> {
        let all_unit_len = all_unit_length(&ca);
        if all_unit_len && self.function_returns_scalar {
            ac.with_agg_state(AggState::AggregatedScalar(
                ca.explode().unwrap().into_series(),
            ));
            ac.with_update_groups(UpdateGroups::No);
        } else {
            ac.with_series(ca.into_series(), true, Some(&self.expr))?;
            ac.with_update_groups(UpdateGroups::WithSeriesLen);
        }

        Ok(ac)
    }

    fn get_input_schema(&self, _df: &DataFrame) -> Cow<Schema> {
        Cow::Borrowed(self.input_schema.as_ref())
    }

    /// Evaluates and flattens `Option<Column>` to `Column`.
    fn eval_and_flatten(&self, inputs: &mut [Column]) -> PolarsResult<Column> {
        if let Some(out) = self.function.call_udf(inputs)? {
            Ok(out)
        } else {
            let field = self.to_field(self.input_schema.as_ref()).unwrap();
            Ok(Column::full_null(field.name().clone(), 1, field.dtype()))
        }
    }
    fn apply_single_group_aware<'a>(
        &self,
        mut ac: AggregationContext<'a>,
    ) -> PolarsResult<AggregationContext<'a>> {
        let s = ac.series();

        polars_ensure!(
            !matches!(ac.agg_state(), AggState::AggregatedScalar(_)),
            expr = self.expr,
            ComputeError: "cannot aggregate, the column is already aggregated",
        );

        let name = s.name().clone();
        let agg = ac.aggregated();
        // Collection of empty list leads to a null dtype. See: #3687.
        if agg.len() == 0 {
            // Create input for the function to determine the output dtype, see #3946.
            let agg = agg.list().unwrap();
            let input_dtype = agg.inner_dtype();
            let input = Column::full_null(PlSmallStr::EMPTY, 0, input_dtype);

            let output = self.eval_and_flatten(&mut [input])?;
            let ca = ListChunked::full(name, output.as_materialized_series(), 0);
            return self.finish_apply_groups(ac, ca);
        }

        let f = |opt_s: Option<Series>| match opt_s {
            None => Ok(None),
            Some(mut s) => {
                if self.pass_name_to_apply {
                    s.rename(name.clone());
                }
                Ok(self
                    .function
                    .call_udf(&mut [Column::from(s)])?
                    .map(|c| c.as_materialized_series().clone()))
            },
        };

        let ca: ListChunked = if self.allow_threading {
            let dtype = if self.output_field.dtype.is_known() && !self.output_field.dtype.is_null()
            {
                Some(self.output_field.dtype.clone())
            } else {
                None
            };

            let lst = agg.list().unwrap();
            let iter = lst.par_iter().map(f);

            if let Some(dtype) = dtype {
                // TODO! uncomment this line and remove debug_assertion after a while.
                // POOL.install(|| {
                //     iter.collect_ca_with_dtype::<PolarsResult<_>>(PlSmallStr::EMPTY, DataType::List(Box::new(dtype)))
                // })?
                let out: ListChunked = POOL.install(|| iter.collect::<PolarsResult<_>>())?;

                debug_assert_eq!(out.dtype(), &DataType::List(Box::new(dtype)));

                out
            } else {
                POOL.install(|| iter.collect::<PolarsResult<_>>())?
            }
        } else {
            agg.list()
                .unwrap()
                .into_iter()
                .map(f)
                .collect::<PolarsResult<_>>()?
        };

        self.finish_apply_groups(ac, ca.with_name(name))
    }

    /// Apply elementwise e.g. ignore the group/list indices.
    fn apply_single_elementwise<'a>(
        &self,
        mut ac: AggregationContext<'a>,
    ) -> PolarsResult<AggregationContext<'a>> {
        let (s, aggregated) = match ac.agg_state() {
            AggState::AggregatedList(s) => {
                let ca = s.list().unwrap();
                let out = ca.apply_to_inner(&|s| {
                    self.eval_and_flatten(&mut [s.into()])
                        .map(|c| c.as_materialized_series().clone())
                })?;
                (out.into_series(), true)
            },
            AggState::NotAggregated(s) => {
                let (out, aggregated) = (
                    self.eval_and_flatten(&mut [s.clone().into()])?
                        .as_materialized_series()
                        .clone(),
                    false,
                );
                check_map_output_len(s.len(), out.len(), &self.expr)?;
                (out, aggregated)
            },
            agg_state => {
                ac.with_agg_state(agg_state.try_map(|s| {
                    self.eval_and_flatten(&mut [s.clone().into()])
                        .map(|c| c.as_materialized_series().clone())
                })?);
                return Ok(ac);
            },
        };

        ac.with_series_and_args(s, aggregated, Some(&self.expr), true)?;
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

        // Aggregate representation of the aggregation contexts,
        // then unpack the lists and finally create iterators from this list chunked arrays.
        let mut iters = acs
            .iter_mut()
            .map(|ac| ac.iter_groups(self.pass_name_to_apply))
            .collect::<Vec<_>>();

        // Length of the items to iterate over.
        let len = iters[0].size_hint().0;

        let ca = if len == 0 {
            let mut builder = get_list_builder(&field.dtype, len * 5, len, field.name);
            for _ in 0..len {
                container.clear();
                for iter in &mut iters {
                    match iter.next().unwrap() {
                        None => {
                            builder.append_null();
                        },
                        Some(s) => container.push(s.deep_clone().into()),
                    }
                }
                let out = self
                    .function
                    .call_udf(&mut container)
                    .map(|r| r.map(|c| c.as_materialized_series().clone()))?;

                builder.append_opt_series(out.as_ref())?
            }
            builder.finish()
        } else {
            // We still need this branch to materialize unknown/ data dependent types in eager. :(
            (0..len)
                .map(|_| {
                    container.clear();
                    for iter in &mut iters {
                        match iter.next().unwrap() {
                            None => return Ok(None),
                            Some(s) => container.push(s.deep_clone().into()),
                        }
                    }
                    self.function
                        .call_udf(&mut container)
                        .map(|r| r.map(|c| c.as_materialized_series().clone()))
                })
                .collect::<PolarsResult<ListChunked>>()?
                .with_name(field.name.clone())
        };
        #[cfg(debug_assertions)]
        {
            let inner = ca.dtype().inner_dtype().unwrap();
            if field.dtype.is_known() {
                assert_eq!(inner, &field.dtype);
            }
        }

        drop(iters);

        // Take the first aggregation context that as that is the input series.
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
        input_len == output_len, expr = expr, InvalidOperation:
        "output length of `map` ({}) must be equal to the input length ({}); \
        consider using `apply` instead", output_len, input_len
    );
    Ok(())
}

impl PhysicalExpr for ApplyExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let f = |e: &Arc<dyn PhysicalExpr>| e.evaluate(df, state);
        let mut inputs = if self.allow_threading && self.inputs.len() > 1 {
            POOL.install(|| {
                self.inputs
                    .par_iter()
                    .map(f)
                    .map(|v| v.map(Column::from))
                    .collect::<PolarsResult<Vec<_>>>()
            })
        } else {
            self.inputs
                .iter()
                .map(f)
                .map(|v| v.map(Column::from))
                .collect::<PolarsResult<Vec<_>>>()
        }?;

        if self.allow_rename {
            self.eval_and_flatten(&mut inputs)
                .map(|c| c.as_materialized_series().clone())
        } else {
            let in_name = inputs[0].name().clone();
            Ok(self
                .eval_and_flatten(&mut inputs)?
                .as_materialized_series()
                .clone()
                .with_name(in_name))
        }
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        polars_ensure!(
            self.allow_group_aware,
            expr = self.expr,
            ComputeError: "this expression cannot run in the group_by context",
        );
        if self.inputs.len() == 1 {
            let mut ac = self.inputs[0].evaluate_on_groups(df, groups, state)?;

            match self.collect_groups {
                ApplyOptions::ApplyList => {
                    let s = self
                        .eval_and_flatten(&mut [ac.aggregated().into()])?
                        .as_materialized_series()
                        .clone();
                    ac.with_series(s, true, Some(&self.expr))?;
                    Ok(ac)
                },
                ApplyOptions::GroupWise => self.apply_single_group_aware(ac),
                ApplyOptions::ElementWise => self.apply_single_elementwise(ac),
            }
        } else {
            let mut acs = self.prepare_multiple_inputs(df, groups, state)?;

            match self.collect_groups {
                ApplyOptions::ApplyList => {
                    let mut s = acs
                        .iter_mut()
                        .map(|ac| ac.aggregated().into())
                        .collect::<Vec<_>>();
                    let s = self
                        .eval_and_flatten(&mut s)?
                        .as_materialized_series()
                        .clone();
                    // take the first aggregation context that as that is the input series
                    let mut ac = acs.swap_remove(0);
                    ac.with_update_groups(UpdateGroups::WithGroupsLen);
                    ac.with_series(s, true, Some(&self.expr))?;
                    Ok(ac)
                },
                ApplyOptions::GroupWise => self.apply_multiple_group_aware(acs, df),
                ApplyOptions::ElementWise => {
                    let mut has_agg_list = false;
                    let mut has_agg_scalar = false;
                    let mut has_not_agg = false;
                    for ac in &acs {
                        match ac.state {
                            AggState::AggregatedList(_) => has_agg_list = true,
                            AggState::AggregatedScalar(_) => has_agg_scalar = true,
                            AggState::NotAggregated(_) => has_not_agg = true,
                            _ => {},
                        }
                    }
                    if has_agg_list || (has_agg_scalar && has_not_agg) {
                        self.apply_multiple_group_aware(acs, df)
                    } else {
                        apply_multiple_elementwise(
                            acs,
                            self.function.as_ref(),
                            &self.expr,
                            self.check_lengths,
                        )
                    }
                },
            }
        }
    }
    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.expr.to_field(input_schema, Context::Default)
    }
    #[cfg(feature = "parquet")]
    fn as_stats_evaluator(&self) -> Option<&dyn StatsEvaluator> {
        let function = match &self.expr {
            Expr::Function { function, .. } => function,
            _ => return None,
        };

        match function {
            FunctionExpr::Boolean(BooleanFunction::IsNull) => Some(self),
            #[cfg(feature = "is_in")]
            FunctionExpr::Boolean(BooleanFunction::IsIn) => Some(self),
            #[cfg(feature = "is_between")]
            FunctionExpr::Boolean(BooleanFunction::IsBetween { closed: _ }) => Some(self),
            FunctionExpr::Boolean(BooleanFunction::IsNotNull) => Some(self),
            _ => None,
        }
    }
    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        if self.inputs.len() == 1 && matches!(self.collect_groups, ApplyOptions::ElementWise) {
            Some(self)
        } else {
            None
        }
    }
    fn is_scalar(&self) -> bool {
        self.function_returns_scalar || self.function_operates_on_scalar
    }
}

fn apply_multiple_elementwise<'a>(
    mut acs: Vec<AggregationContext<'a>>,
    function: &dyn ColumnsUdf,
    expr: &Expr,
    check_lengths: bool,
) -> PolarsResult<AggregationContext<'a>> {
    match acs.first().unwrap().agg_state() {
        // A fast path that doesn't drop groups of the first arg.
        // This doesn't require group re-computation.
        AggState::AggregatedList(s) => {
            let ca = s.list().unwrap();

            let other = acs[1..]
                .iter()
                .map(|ac| ac.flat_naive().into_owned().into())
                .collect::<Vec<_>>();

            let out = ca.apply_to_inner(&|s| {
                let mut args = Vec::with_capacity(other.len() + 1);
                args.push(s.into());
                args.extend_from_slice(&other);
                Ok(function
                    .call_udf(&mut args)?
                    .unwrap()
                    .as_materialized_series()
                    .clone())
            })?;
            let mut ac = acs.swap_remove(0);
            ac.with_series(out.into_series(), true, None)?;
            Ok(ac)
        },
        first_as => {
            let check_lengths = check_lengths && !matches!(first_as, AggState::Literal(_));
            let aggregated = acs.iter().all(|ac| ac.is_aggregated() | ac.is_literal())
                && acs.iter().any(|ac| ac.is_aggregated());
            let mut s = acs
                .iter_mut()
                .enumerate()
                .map(|(i, ac)| {
                    // Make sure the groups are updated because we are about to throw away
                    // the series length information, only on the first iteration.
                    if let (0, UpdateGroups::WithSeriesLen) = (i, &ac.update_groups) {
                        ac.groups();
                    }

                    ac.flat_naive().into_owned()
                })
                .map(Column::from)
                .collect::<Vec<_>>();

            let input_len = s[0].len();
            let s = function
                .call_udf(&mut s)?
                .unwrap()
                .as_materialized_series()
                .clone();
            if check_lengths {
                check_map_output_len(input_len, s.len(), expr)?;
            }

            // Take the first aggregation context that as that is the input series.
            let mut ac = acs.swap_remove(0);
            ac.with_series_and_args(s, aggregated, None, true)?;
            Ok(ac)
        },
    }
}

#[cfg(feature = "parquet")]
impl StatsEvaluator for ApplyExpr {
    fn should_read(&self, stats: &BatchStats) -> PolarsResult<bool> {
        let read = self.should_read_impl(stats)?;
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
        // Ensure the input of the function is only a `col(..)`.
        // If it does any arithmetic the code below is flawed.
        if !matches!(input[0], Expr::Column(_)) {
            return Ok(true);
        }

        match function {
            FunctionExpr::Boolean(BooleanFunction::IsNull) => {
                let root = expr_to_leaf_column_name(&self.expr)?;

                match stats.get_stats(&root).ok() {
                    Some(st) => match st.null_count() {
                        Some(0) => Ok(false),
                        _ => Ok(true),
                    },
                    None => Ok(true),
                }
            },
            FunctionExpr::Boolean(BooleanFunction::IsNotNull) => {
                let root = expr_to_leaf_column_name(&self.expr)?;

                match stats.get_stats(&root).ok() {
                    Some(st) => match st.null_count() {
                        Some(null_count)
                            if stats
                                .num_rows()
                                .map_or(false, |num_rows| num_rows == null_count) =>
                        {
                            Ok(false)
                        },
                        _ => Ok(true),
                    },
                    None => Ok(true),
                }
            },
            #[cfg(feature = "is_in")]
            FunctionExpr::Boolean(BooleanFunction::IsIn) => {
                let should_read = || -> Option<bool> {
                    let root = expr_to_leaf_column_name(&input[0]).ok()?;
                    let Expr::Literal(LiteralValue::Series(input)) = &input[1] else {
                        return None;
                    };
                    #[allow(clippy::explicit_auto_deref)]
                    let input: &Series = &**input;
                    let st = stats.get_stats(&root).ok()?;
                    let min = st.to_min()?;
                    let max = st.to_max()?;

                    if max.get(0).unwrap() == min.get(0).unwrap() {
                        let one_equals =
                            |value: &Series| Some(ChunkCompareEq::equal(input, value).ok()?.any());
                        return one_equals(min);
                    }

                    let smaller = ChunkCompareIneq::lt(input, min).ok()?;
                    let bigger = ChunkCompareIneq::gt(input, max).ok()?;

                    Some(!(smaller | bigger).all())
                };

                Ok(should_read().unwrap_or(true))
            },
            #[cfg(feature = "is_between")]
            FunctionExpr::Boolean(BooleanFunction::IsBetween { closed }) => {
                let should_read = || -> Option<bool> {
                    let root: PlSmallStr = expr_to_leaf_column_name(&input[0]).ok()?;
                    let Expr::Literal(left) = &input[1] else {
                        return None;
                    };
                    let Expr::Literal(right) = &input[2] else {
                        return None;
                    };

                    let st = stats.get_stats(&root).ok()?;
                    let min = st.to_min()?;
                    let max = st.to_max()?;

                    let (left, left_dtype) = (left.to_any_value()?, left.get_datatype());
                    let (right, right_dtype) = (right.to_any_value()?, right.get_datatype());

                    let left = Series::from_any_values_and_dtype(
                        PlSmallStr::EMPTY,
                        &[left],
                        &left_dtype,
                        false,
                    )
                    .ok()?;
                    let right = Series::from_any_values_and_dtype(
                        PlSmallStr::EMPTY,
                        &[right],
                        &right_dtype,
                        false,
                    )
                    .ok()?;

                    // don't read the row_group anyways as
                    // the condition will evaluate to false.
                    // e.g. in_between(10, 5)
                    if ChunkCompareIneq::gt(&left, &right).ok()?.all() {
                        return Some(false);
                    }

                    let (left_open, right_open) = match closed {
                        ClosedInterval::None => (true, true),
                        ClosedInterval::Both => (false, false),
                        ClosedInterval::Left => (false, true),
                        ClosedInterval::Right => (true, false),
                    };
                    // check the right limit of the interval.
                    // if the end is open, we should be stricter (lt_eq instead of lt).
                    if right_open && ChunkCompareIneq::lt_eq(&right, min).ok()?.all()
                        || !right_open && ChunkCompareIneq::lt(&right, min).ok()?.all()
                    {
                        return Some(false);
                    }
                    // we couldn't conclude anything using the right limit,
                    // check the left limit of the interval
                    if left_open && ChunkCompareIneq::gt_eq(&left, max).ok()?.all()
                        || !left_open && ChunkCompareIneq::gt(&left, max).ok()?.all()
                    {
                        return Some(false);
                    }
                    // read the row_group
                    Some(true)
                };

                Ok(should_read().unwrap_or(true))
            },
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
        let s = a.evaluate_partitioned(df, groups, state)?.into();

        if self.allow_rename {
            self.eval_and_flatten(&mut [s])
                .map(|c| c.as_materialized_series().clone())
        } else {
            let in_name = s.name().clone();
            Ok(self
                .eval_and_flatten(&mut [s])?
                .as_materialized_series()
                .clone()
                .with_name(in_name))
        }
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
