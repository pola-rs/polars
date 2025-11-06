use std::borrow::Cow;

use polars_core::POOL;
use polars_core::chunked_array::builder::get_list_builder;
use polars_core::chunked_array::from_iterator_par::{
    ChunkedCollectParIterExt, try_list_from_par_iter,
};
use polars_core::prelude::*;
use rayon::prelude::*;

use super::*;
use crate::dispatch::GroupsUdf;
use crate::expressions::{AggState, AggregationContext, PhysicalExpr, UpdateGroups};

#[derive(Clone)]
pub struct ApplyExpr {
    inputs: Vec<Arc<dyn PhysicalExpr>>,
    function: SpecialEq<Arc<dyn ColumnsUdf>>,
    groups_function: Option<SpecialEq<Arc<dyn GroupsUdf>>>,
    expr: Expr,
    flags: FunctionFlags,
    function_operates_on_scalar: bool,
    input_schema: SchemaRef,
    allow_threading: bool,
    check_lengths: bool,
    is_fallible: bool,

    /// Output field of the expression excluding potential aggregation.
    output_field: Field,
}

impl ApplyExpr {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        inputs: Vec<Arc<dyn PhysicalExpr>>,
        function: SpecialEq<Arc<dyn ColumnsUdf>>,
        groups_function: Option<SpecialEq<Arc<dyn GroupsUdf>>>,
        expr: Expr,
        options: FunctionOptions,
        allow_threading: bool,
        input_schema: SchemaRef,
        non_aggregated_output_field: Field,
        function_operates_on_scalar: bool,
        is_fallible: bool,
    ) -> Self {
        debug_assert!(
            !options.is_length_preserving()
                || !options.flags.contains(FunctionFlags::RETURNS_SCALAR),
            "expr {expr:?} is not implemented correctly. 'returns_scalar' and 'elementwise' are mutually exclusive",
        );

        Self {
            inputs,
            function,
            groups_function,
            expr,
            flags: options.flags,
            function_operates_on_scalar,
            input_schema,
            allow_threading,
            check_lengths: options.check_lengths(),
            output_field: non_aggregated_output_field,
            is_fallible,
        }
    }

    #[allow(clippy::ptr_arg)]
    fn prepare_multiple_inputs<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
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
        let c = if self.is_scalar() {
            let out = ca.explode(false).unwrap();
            // if the explode doesn't return the same len, it wasn't scalar.
            polars_ensure!(out.len() == ca.len(), InvalidOperation: "expected scalar for expr: {}, got {}", self.expr, &out);
            ac.update_groups = UpdateGroups::No;
            out.into_column()
        } else {
            ac.with_update_groups(UpdateGroups::WithSeriesLen);
            ca.into_series().into()
        };

        ac.with_values_and_args(c, true, None, false, self.is_scalar())?;

        Ok(ac)
    }

    fn get_input_schema(&self, _df: &DataFrame) -> Cow<'_, Schema> {
        Cow::Borrowed(self.input_schema.as_ref())
    }

    /// Evaluates and flattens `Option<Column>` to `Column`.
    fn eval_and_flatten(&self, inputs: &mut [Column]) -> PolarsResult<Column> {
        self.function.call_udf(inputs)
    }

    fn apply_single_group_aware<'a>(
        &self,
        mut ac: AggregationContext<'a>,
    ) -> PolarsResult<AggregationContext<'a>> {
        // Fix up groups for AggregatedScalar, so that we can pretend they are just normal groups.
        ac.set_groups_for_undefined_agg_states();

        let name = ac.get_values().name().clone();
        let f = |opt_s: Option<Series>| match opt_s {
            None => Ok(None),
            Some(mut s) => {
                if self.flags.contains(FunctionFlags::PASS_NAME_TO_APPLY) {
                    s.rename(name.clone());
                }
                Ok(Some(
                    self.function
                        .call_udf(&mut [Column::from(s)])?
                        .take_materialized_series(),
                ))
            },
        };

        // In case of overlapping (rolling) groups, we build groups in a lazy manner to avoid
        // memory explosion.
        // TODO: Add parallel iterator path; support Idx GroupsType.
        if matches!(ac.agg_state(), AggState::NotAggregated(_))
            && let GroupsType::Slice {
                overlapping: true, ..
            } = ac.groups.as_ref().as_ref()
        {
            let ca: ChunkedArray<_> = ac
                .iter_groups_lazy(false)
                .map(|opt| opt.map(|s| s.as_ref().clone()))
                .map(f)
                .collect::<PolarsResult<_>>()?;

            return self.finish_apply_groups(ac, ca.with_name(name));
        }

        // At this point, calling aggregated() will not lead to memory explosion.
        let agg = match ac.agg_state() {
            AggState::AggregatedScalar(s) => s.as_list().into_column(),
            _ => ac.aggregated(),
        };

        // Collection of empty list leads to a null dtype. See: #3687.
        if agg.is_empty() {
            // Create input for the function to determine the output dtype, see #3946.
            let agg = agg.list().unwrap();
            let input_dtype = agg.inner_dtype();
            let input = Column::full_null(name.clone(), 0, input_dtype);

            let output = self.eval_and_flatten(&mut [input])?;
            let ca = ListChunked::full(name, output.as_materialized_series(), 0);
            return self.finish_apply_groups(ac, ca);
        }

        let ca: ListChunked = if self.allow_threading {
            let lst = agg.list().unwrap();
            let iter = lst.par_iter().map(f);

            if self.output_field.dtype.is_known() {
                let dtype = self.output_field.dtype.clone();
                let dtype = dtype.implode();
                POOL.install(|| {
                    iter.collect_ca_with_dtype::<PolarsResult<_>>(PlSmallStr::EMPTY, dtype)
                })?
            } else {
                POOL.install(|| try_list_from_par_iter(iter, PlSmallStr::EMPTY))?
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
        let (c, aggregated) = match ac.agg_state() {
            AggState::AggregatedList(c) => {
                let ca = c.list().unwrap();
                let out = ca.apply_to_inner(&|s| {
                    Ok(self
                        .eval_and_flatten(&mut [s.into_column()])?
                        .take_materialized_series())
                })?;
                (out.into_column(), true)
            },
            AggState::NotAggregated(c) => {
                let (out, aggregated) = (self.eval_and_flatten(&mut [c.clone()])?, false);
                check_map_output_len(c.len(), out.len(), &self.expr)?;
                (out, aggregated)
            },
            agg_state => {
                ac.with_agg_state(agg_state.try_map(|s| self.eval_and_flatten(&mut [s.clone()]))?);
                return Ok(ac);
            },
        };

        ac.with_values_and_args(c, aggregated, Some(&self.expr), true, self.is_scalar())?;
        Ok(ac)
    }

    // Fast-path when every AggState is a LiteralScalar. This path avoids calling aggregated() or
    // groups(), and returns a LiteralScalar, on the implicit condition that the function is pure.
    fn apply_all_literal_elementwise<'a>(
        &self,
        mut acs: Vec<AggregationContext<'a>>,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut cols = acs
            .iter()
            .map(|ac| ac.get_values().clone())
            .collect::<Vec<_>>();
        let out = self.function.call_udf(&mut cols)?;
        polars_ensure!(
            out.len() == 1,
            ComputeError: "elementwise expression {:?} must return exactly 1 value on literals, got {}",
                &self.expr, out.len()
        );
        let mut ac = acs.pop().unwrap();
        ac.with_literal(out);
        Ok(ac)
    }

    fn apply_multiple_elementwise<'a>(
        &self,
        mut acs: Vec<AggregationContext<'a>>,
        must_aggregate: bool,
    ) -> PolarsResult<AggregationContext<'a>> {
        // At this stage, we either have (with or without LiteralScalars):
        // - one or more AggregatedList or NotAggregated ACs
        // - one or more AggregatedScalar ACs

        let mut previous = None;
        for ac in acs.iter_mut() {
            // TBD: If we want to be strict, we would check all groups
            if matches!(
                ac.state,
                AggState::LiteralScalar(_) | AggState::AggregatedScalar(_)
            ) {
                continue;
            }

            if must_aggregate {
                ac.aggregated();
            }

            if matches!(ac.state, AggState::AggregatedList(_)) {
                if let Some(p) = previous {
                    ac.groups().check_lengths(p)?;
                }
                previous = Some(ac.groups());
            }
        }

        // At this stage, we do not have both AggregatedList and NotAggregated ACs

        // The first non-LiteralScalar AC will be used as the base AC to retain the context
        let base_ac_idx = acs.iter().position(|ac| !ac.is_literal()).unwrap();

        match acs[base_ac_idx].agg_state() {
            AggState::AggregatedList(s) => {
                let aggregated = acs.iter().any(|ac| ac.is_aggregated());
                let ca = s.list().unwrap();
                let input_len = s.len();

                let out = ca.apply_to_inner(&|_| {
                    let mut cols = acs
                        .iter()
                        .map(|ac| ac.flat_naive().into_owned())
                        .collect::<Vec<_>>();
                    Ok(self
                        .function
                        .call_udf(&mut cols)?
                        .as_materialized_series()
                        .clone())
                })?;

                let out = out.into_column();
                if self.check_lengths {
                    check_map_output_len(input_len, out.len(), &self.expr)?;
                }

                let mut ac = acs.swap_remove(base_ac_idx);
                ac.with_values_and_args(
                    out,
                    aggregated,
                    Some(&self.expr),
                    false,
                    self.is_scalar(),
                )?;
                Ok(ac)
            },
            _ => {
                let aggregated = acs.iter().any(|ac| ac.is_aggregated());
                debug_assert!(aggregated == self.is_scalar());

                let mut cols = acs
                    .iter()
                    .map(|ac| ac.flat_naive().into_owned())
                    .collect::<Vec<_>>();

                let input_len = cols[base_ac_idx].len();
                let out = self.function.call_udf(&mut cols)?;
                if self.check_lengths {
                    check_map_output_len(input_len, out.len(), &self.expr)?;
                }

                let mut ac = acs.swap_remove(base_ac_idx);
                ac.with_values_and_args(
                    out,
                    aggregated,
                    Some(&self.expr),
                    false,
                    self.is_scalar(),
                )?;
                Ok(ac)
            },
        }
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
            .map(|ac| ac.iter_groups(self.flags.contains(FunctionFlags::PASS_NAME_TO_APPLY)))
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
                    .map(|c| c.take_materialized_series())?;

                builder.append_series(&out)?
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
                    Ok(Some(
                        self.function
                            .call_udf(&mut container)?
                            .take_materialized_series(),
                    ))
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

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
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

        if self.flags.contains(FunctionFlags::ALLOW_RENAME) {
            self.eval_and_flatten(&mut inputs)
        } else {
            let in_name = inputs[0].name().clone();
            Ok(self.eval_and_flatten(&mut inputs)?.with_name(in_name))
        }
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        // Some function have specialized implementation.
        if let Some(groups_function) = self.groups_function.as_ref() {
            return groups_function.evaluate_on_groups(&self.inputs, df, groups, state);
        }

        if self.inputs.len() == 1 {
            let mut ac = self.inputs[0].evaluate_on_groups(df, groups, state)?;

            if self.flags.is_elementwise() && (!self.is_fallible || ac.groups_cover_all_values()) {
                self.apply_single_elementwise(ac)
            } else {
                self.apply_single_group_aware(ac)
            }
        } else {
            let mut acs = self.prepare_multiple_inputs(df, groups, state)?;

            match self.flags.is_elementwise() {
                false => self.apply_multiple_group_aware(acs, df),
                true => {
                    // Implementation dispatch:
                    // The current implementation of `apply_multiple_elementwise` requires the
                    // multiple inputs to have a compatible data layout as it invokes `flat_naive()`.
                    // Compatible means matching as-is, or possibly matching after aggregation,
                    // or matching after an implicit broadcast by the function.

                    // The dispatch logic between the implementations depends on the combination of aggstates:
                    // - Any presence of LiteralScalar is immaterial as it gets broadcasted in the UDF.
                    // - Combination of AggregatedScalar and AggregatedList => NOT compatible.
                    // - Combination of AggregatedScalar and NotAggregated => NOT compatible.
                    // - Any other combination => comptable, and thereforee allowed for elementwise.
                    //   In this case, aggregated() on NotAggregated may be required; however, it can be
                    //   prohibitively memory expensive when dealing with overlapping (e.g., rolling) groups,
                    //   in which case we fall-back to group_aware.

                    // Consequently, these may follow the elementwise path (not exhaustive):
                    // - All AggregatedScalar
                    // - A combination of AggregatedList(s) and NotAggregated(s) without expensive aggregation.
                    // - Either of the above with or without LiteralScalar

                    // Visually, in the case of 2 aggstates:
                    // Legend:
                    // - el = elementwise, no need to aggregate() NotAgg
                    // - el + agg = elementwise, but must aggregate() NotAgg
                    // - ga = group_aware
                    // - alit = all_literal
                    // - * = broadcast falls back to group_aware
                    // - ~ = same a smirror pair (symmetric)
                    //
                    //              | AggList | NotAgg   | AggScalar | LitScalar
                    //   --------------------------------------------------------
                    //    AggList   |   el*   | depends* |    ga     |     el
                    //    NotAgg    |    ~    | depends* |    ga     |     el
                    //    AggScalar |    ~    |    ~     |    el     |     el
                    //    LitScalar |    ~    |    ~     |     ~     |    alit
                    //
                    // In case it depends, extending to any combination of multiple aggstates
                    // (a) Multiple NotAggs, w/o AggList
                    //
                    //                   | !has_rolling | has_rolling
                    //   -------------------------------------------------
                    //    groups match   |      el      |     el
                    //    groups_diverge |    el+agg    |     ga
                    //
                    // (b) Multiple NotAggs, with at least 1 AggList
                    //
                    //                   | !has_rolling | has_rolling
                    //   -------------------------------------------------
                    //    groups match   |    el+agg    |     ga
                    //    groups diverge |    el+agg    |     ga
                    //
                    //  * Finally, when broadcast is required in non-scalar we switch to group_aware

                    // Collect statistics on input aggstates
                    let mut has_agg_list = false;
                    let mut has_agg_scalar = false;
                    let mut has_not_agg = false;
                    let mut has_not_agg_with_overlapping_groups = false;
                    let mut not_agg_groups_may_diverge = false;

                    let mut previous: Option<&AggregationContext<'_>> = None;
                    for ac in &acs {
                        match ac.state {
                            AggState::AggregatedList(_) => {
                                has_agg_list = true;
                            },
                            AggState::AggregatedScalar(_) => has_agg_scalar = true,
                            AggState::NotAggregated(_) => {
                                has_not_agg = true;
                                if let Some(p) = previous {
                                    not_agg_groups_may_diverge |=
                                        !std::ptr::eq(p.groups.as_ref(), ac.groups.as_ref());
                                }
                                previous = Some(ac);
                                if ac.groups.is_overlapping() {
                                    has_not_agg_with_overlapping_groups = true;
                                }
                            },
                            _ => {},
                        }
                    }

                    let all_literal = !(has_agg_list || has_agg_scalar || has_not_agg);
                    let elementwise_must_aggregate =
                        has_not_agg && (has_agg_list || not_agg_groups_may_diverge);

                    if all_literal {
                        // Fast path
                        self.apply_all_literal_elementwise(acs)
                    } else if has_agg_scalar && (has_agg_list || has_not_agg) {
                        // Not compatible
                        self.apply_multiple_group_aware(acs, df)
                    } else if elementwise_must_aggregate && has_not_agg_with_overlapping_groups {
                        // Compatible but calling aggregated() is too expensive
                        self.apply_multiple_group_aware(acs, df)
                    } else if self.is_fallible
                        && acs.iter_mut().any(|ac| !ac.groups_cover_all_values())
                    {
                        // Fallible expression and there are elements that are masked out.
                        self.apply_multiple_group_aware(acs, df)
                    } else {
                        // Broadcast in NotAgg or AggList requires group_aware
                        acs.iter_mut().filter(|ac| !ac.is_literal()).for_each(|ac| {
                            ac.groups();
                        });
                        let has_broadcast =
                            if let Some(base_ac_idx) = acs.iter().position(|ac| !ac.is_literal()) {
                                acs.iter()
                                    .enumerate()
                                    .filter(|(i, ac)| *i != base_ac_idx && !ac.is_literal())
                                    .any(|(_, ac)| {
                                        acs[base_ac_idx].groups.iter().zip(ac.groups.iter()).any(
                                            |(l, r)| {
                                                l.len() != r.len() && (l.len() == 1 || r.len() == 1)
                                            },
                                        )
                                    })
                            } else {
                                false
                            };
                        if has_broadcast {
                            //  Broadcast fall-back.
                            self.apply_multiple_group_aware(acs, df)
                        } else {
                            self.apply_multiple_elementwise(acs, elementwise_must_aggregate)
                        }
                    }
                },
            }
        }
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }
    fn is_scalar(&self) -> bool {
        self.flags.returns_scalar()
            || (self.function_operates_on_scalar && self.flags.is_length_preserving())
    }
}
