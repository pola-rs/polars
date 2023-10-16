use std::borrow::Cow;
use std::sync::Arc;

use polars_core::frame::group_by::GroupsProxy;
use polars_core::prelude::*;
use polars_core::POOL;
#[cfg(feature = "parquet")]
use polars_io::predicates::{BatchStats, StatsEvaluator};
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
    pub allow_threading: bool,
    pub check_lengths: bool,
    pub allow_group_aware: bool,
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
            allow_threading: true,
            check_lengths: true,
            allow_group_aware: true,
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

    /// Evaluates and flattens `Option<Series>` to `Series`.
    fn eval_and_flatten(&self, inputs: &mut [Series]) -> PolarsResult<Series> {
        if let Some(out) = self.function.call_udf(inputs)? {
            Ok(out)
        } else {
            let field = self.to_field(self.input_schema.as_ref().unwrap()).unwrap();
            Ok(Series::full_null(field.name(), 1, field.data_type()))
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

        let name = s.name().to_string();
        let agg = ac.aggregated();
        // Collection of empty list leads to a null dtype. See: #3687.
        if agg.len() == 0 {
            // Create input for the function to determine the output dtype, see #3946.
            let agg = agg.list().unwrap();
            let input_dtype = agg.inner_dtype();
            let input = Series::full_null("", 0, &input_dtype);

            let output = self.eval_and_flatten(&mut [input])?;
            let ca = ListChunked::full(&name, &output, 0);
            return self.finish_apply_groups(ac, ca);
        }

        let f = |opt_s: Option<Series>| match opt_s {
            None => Ok(None),
            Some(mut s) => {
                if self.pass_name_to_apply {
                    s.rename(&name);
                }
                self.function.call_udf(&mut [s])
            },
        };

        let ca: ListChunked = if self.allow_threading {
            POOL.install(|| {
                agg.list()
                    .unwrap()
                    .par_iter()
                    .map(f)
                    .collect::<PolarsResult<_>>()
            })?
        } else {
            agg.list()
                .unwrap()
                .into_iter()
                .map(f)
                .collect::<PolarsResult<_>>()?
        };

        self.finish_apply_groups(ac, ca.with_name(&name))
    }

    /// Apply elementwise e.g. ignore the group/list indices.
    fn apply_single_elementwise<'a>(
        &self,
        mut ac: AggregationContext<'a>,
    ) -> PolarsResult<AggregationContext<'a>> {
        let (s, aggregated) = match ac.agg_state() {
            AggState::AggregatedList(s) => {
                let ca = s.list().unwrap();
                let out = ca.apply_to_inner(&|s| self.eval_and_flatten(&mut [s]))?;
                (out.into_series(), true)
            },
            AggState::AggregatedScalar(s) => (self.eval_and_flatten(&mut [s.clone()])?, true),
            AggState::NotAggregated(s) | AggState::Literal(s) => {
                let (out, aggregated) = (self.eval_and_flatten(&mut [s.clone()])?, false);
                check_map_output_len(s.len(), out.len(), &self.expr)?;
                (out, aggregated)
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
            .map(|ac|
                // SAFETY: unstable series never lives longer than the iterator.
                unsafe { ac.iter_groups(self.pass_name_to_apply) })
            .collect::<Vec<_>>();

        // Length of the items to iterate over.
        let len = iters[0].size_hint().0;

        if len == 0 {
            let out = Series::full_null(field.name(), 0, &field.dtype);
            drop(iters);

            // Take the first aggregation context that as that is the input series.
            let mut ac = acs.swap_remove(0);
            ac.with_series(out, true, Some(&self.expr))?;
            return Ok(ac);
        }

        let ca = (0..len)
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
            .collect::<PolarsResult<ListChunked>>()?
            .with_name(&field.name);

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
        consider using `apply` instead", input_len, output_len
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
                    .collect::<PolarsResult<Vec<_>>>()
            })
        } else {
            self.inputs.iter().map(f).collect::<PolarsResult<Vec<_>>>()
        }?;

        if self.allow_rename {
            self.eval_and_flatten(&mut inputs)
        } else {
            let in_name = inputs[0].name().to_string();
            Ok(self.eval_and_flatten(&mut inputs)?.with_name(&in_name))
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
                    let s = self.eval_and_flatten(&mut [ac.aggregated()])?;
                    ac.with_series(s, true, Some(&self.expr))?;
                    Ok(ac)
                },
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
                },
                ApplyOptions::ApplyGroups => self.apply_multiple_group_aware(acs, df),
                ApplyOptions::ApplyFlat => {
                    if acs
                        .iter()
                        .any(|ac| matches!(ac.agg_state(), AggState::AggregatedList(_)))
                    {
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
            FunctionExpr::Boolean(BooleanFunction::IsNull) => Some(self),
            #[cfg(feature = "is_in")]
            FunctionExpr::Boolean(BooleanFunction::IsIn) => Some(self),
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
    check_lengths: bool,
) -> PolarsResult<AggregationContext<'a>> {
    match acs.first().unwrap().agg_state() {
        // A fast path that doesn't drop groups of the first arg.
        // This doesn't require group re-computation.
        AggState::AggregatedList(s) => {
            let ca = s.list().unwrap();

            let other = acs[1..]
                .iter()
                .map(|ac| ac.flat_naive().into_owned())
                .collect::<Vec<_>>();

            let out = ca.apply_to_inner(&|s| {
                let mut args = Vec::with_capacity(other.len() + 1);
                args.push(s);
                args.extend_from_slice(&other);
                Ok(function.call_udf(&mut args)?.unwrap())
            })?;
            let mut ac = acs.swap_remove(0);
            ac.with_series(out.into_series(), true, None)?;
            Ok(ac)
        },
        _ => {
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
                .collect::<Vec<_>>();

            let input_len = s[0].len();
            let s = function.call_udf(&mut s)?.unwrap();
            if check_lengths {
                check_map_output_len(input_len, s.len(), expr)?;
            }

            // Take the first aggregation context that as that is the input series.
            let mut ac = acs.swap_remove(0);
            ac.with_series_and_args(s, false, None, true)?;
            Ok(ac)
        },
    }
}

#[cfg(feature = "parquet")]
impl StatsEvaluator for ApplyExpr {
    fn should_read(&self, stats: &BatchStats) -> PolarsResult<bool> {
        let read = self.should_read_impl(stats)?;
        if ExecutionState::new().verbose() {
            if read {
                eprintln!("parquet file must be read, statistics not sufficient for predicate.")
            } else {
                eprintln!("parquet file can be skipped, the statistics were sufficient to apply the predicate.")
            }
        }

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

                    let all_smaller = || Some(ChunkCompare::lt(input, min).ok()?.all());
                    let all_bigger = || Some(ChunkCompare::gt(input, max).ok()?.all());
                    Some(!all_smaller()? && !all_bigger()?)
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
        let s = a.evaluate_partitioned(df, groups, state)?;

        if self.allow_rename {
            self.eval_and_flatten(&mut [s])
        } else {
            let in_name = s.name().to_string();
            Ok(self.eval_and_flatten(&mut [s])?.with_name(&in_name))
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
