use std::sync::Arc;

use polars_arrow::utils::CustomIterTools;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use polars_core::POOL;

use crate::physical_plan::expression_err;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct TernaryExpr {
    predicate: Arc<dyn PhysicalExpr>,
    truthy: Arc<dyn PhysicalExpr>,
    falsy: Arc<dyn PhysicalExpr>,
    expr: Expr,
}

impl TernaryExpr {
    pub fn new(
        predicate: Arc<dyn PhysicalExpr>,
        truthy: Arc<dyn PhysicalExpr>,
        falsy: Arc<dyn PhysicalExpr>,
        expr: Expr,
    ) -> Self {
        Self {
            predicate,
            truthy,
            falsy,
            expr,
        }
    }
}

fn expand_lengths(truthy: &mut Series, falsy: &mut Series, mask: &mut BooleanChunked) {
    let len = std::cmp::max(std::cmp::max(truthy.len(), falsy.len()), mask.len());
    if len > 1 {
        if falsy.len() == 1 {
            *falsy = falsy.new_from_index(0, len);
        }
        if truthy.len() == 1 {
            *truthy = truthy.new_from_index(0, len);
        }
        if mask.len() == 1 {
            *mask = mask.new_from_index(0, len);
        }
    }
}

fn finish_as_iters<'a>(
    mut ac_truthy: AggregationContext<'a>,
    mut ac_falsy: AggregationContext<'a>,
    mut ac_mask: AggregationContext<'a>,
) -> PolarsResult<AggregationContext<'a>> {
    let mut ca: ListChunked = ac_truthy
        .iter_groups()
        .zip(ac_falsy.iter_groups())
        .zip(ac_mask.iter_groups())
        .map(|((truthy, falsy), mask)| {
            match (truthy, falsy, mask) {
                (Some(truthy), Some(falsy), Some(mask)) => Some(
                    truthy
                        .as_ref()
                        .zip_with(mask.as_ref().bool()?, falsy.as_ref()),
                ),
                _ => None,
            }
            .transpose()
        })
        .collect::<PolarsResult<_>>()?;

    ca.rename(ac_truthy.series().name());
    // aggregation leaves only a single chunks
    let arr = ca.downcast_iter().next().unwrap();
    let list_vals_len = arr.values().len();
    let mut out = ca.into_series();

    if ac_truthy.arity_should_explode() && ac_falsy.arity_should_explode() && ac_mask.arity_should_explode() &&
        // exploded list should be equal to groups length
        list_vals_len == ac_truthy.groups.len()
    {
        out = out.explode()?
    }

    ac_truthy.with_series(out, true);
    Ok(ac_truthy)
}

impl PhysicalExpr for TernaryExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let mut state = state.split();
        // don't cache window functions as they run in parallel
        state.remove_cache_window_flag();
        let mask_series = self.predicate.evaluate(df, &state)?;
        let mut mask = mask_series.bool()?.clone();

        let op_truthy = || self.truthy.evaluate(df, &state);
        let op_falsy = || self.falsy.evaluate(df, &state);

        let (truthy, falsy) = POOL.install(|| rayon::join(op_truthy, op_falsy));
        let mut truthy = truthy?;
        let mut falsy = falsy?;

        if truthy.is_empty() {
            return Ok(truthy);
        }
        if falsy.is_empty() {
            return Ok(falsy);
        }
        if mask.is_empty() {
            return Ok(Series::new_empty(truthy.name(), truthy.dtype()));
        }

        expand_lengths(&mut truthy, &mut falsy, &mut mask);

        truthy.zip_with(&mask, &falsy)
    }
    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.truthy.to_field(input_schema)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let aggregation_predicate = self.predicate.is_valid_aggregation();
        if !aggregation_predicate {
            // unwrap will not fail as it is not an aggregation expression.
            eprintln!(
                "The predicate '{}' in 'when->then->otherwise' is not a valid aggregation and might produce a different number of rows than the groupby operation would. This behavior is experimental and may be subject to change", self.predicate.as_expression().unwrap()
            )
        }
        let op_mask = || self.predicate.evaluate_on_groups(df, groups, state);
        let op_truthy = || self.truthy.evaluate_on_groups(df, groups, state);
        let op_falsy = || self.falsy.evaluate_on_groups(df, groups, state);

        let (ac_mask, (ac_truthy, ac_falsy)) =
            POOL.install(|| rayon::join(op_mask, || rayon::join(op_truthy, op_falsy)));

        let ac_mask = ac_mask?;
        let mut ac_truthy = ac_truthy?;
        let mut ac_falsy = ac_falsy?;

        let mask_s = ac_mask.flat_naive();

        // BIG TODO: find which branches are never hit and remove them
        use AggState::*;
        match (ac_truthy.agg_state(), ac_falsy.agg_state()) {
            // all branches are aggregated-flat or literal
            // mask -> aggregated-flat
            // truthy -> aggregated-flat | literal
            // falsy -> aggregated-flat | literal
            // simply align lengths and zip
            (Literal(truthy) | AggregatedFlat(truthy), AggregatedFlat(falsy) | Literal(falsy))
            | (AggregatedList(truthy), AggregatedList(falsy))
                if matches!(ac_mask.agg_state(), AggState::AggregatedFlat(_)) =>
            {
                let mut truthy = truthy.clone();
                let mut falsy = falsy.clone();
                let mut mask = ac_mask.series().bool()?.clone();
                expand_lengths(&mut truthy, &mut falsy, &mut mask);
                let mut out = truthy.zip_with(&mask, &falsy).unwrap();
                out.rename(truthy.name());
                ac_truthy.with_series(out, true);
                Ok(ac_truthy)
            }

            // we cannot flatten a list because that changes the order, so we apply over groups
            (AggregatedList(_), NotAggregated(_)) | (NotAggregated(_), AggregatedList(_)) => {
                finish_as_iters(ac_truthy, ac_falsy, ac_mask)
            }
            // then:
            //     col().shift()
            // otherwise:
            //     None
            (AggregatedList(_), Literal(_)) | (Literal(_), AggregatedList(_)) => {
                if !aggregation_predicate {
                    // experimental elementwise behavior tested in `test_binary_agg_context_1`
                    return finish_as_iters(ac_truthy, ac_falsy, ac_mask);
                }
                let mask = mask_s.bool()?;
                let check_length = |ca: &ListChunked, mask: &BooleanChunked| {
                    if ca.len() != mask.len() {
                        let msg = format!("The predicates length: '{}' does not match the length of the groups: {}.", mask.len(), ca.len());
                        Err(expression_err!(msg, self.expr, ComputeError))
                    } else {
                        Ok(())
                    }
                };

                if ac_falsy.is_literal() && self.falsy.as_expression().map(has_null) == Some(true) {
                    let s = ac_truthy.aggregated();
                    let ca = s.list().unwrap();
                    check_length(ca, mask)?;
                    let mut out: ListChunked = ca
                        .into_iter()
                        .zip(mask.into_iter())
                        .map(|(truthy, take)| match (truthy, take) {
                            (Some(v), Some(true)) => Some(v),
                            (Some(_), Some(false)) => None,
                            _ => None,
                        })
                        .collect_trusted();
                    out.rename(ac_truthy.series().name());
                    ac_truthy.with_series(out.into_series(), true);
                    Ok(ac_truthy)
                } else if ac_truthy.is_literal()
                    && self.truthy.as_expression().map(has_null) == Some(true)
                {
                    let s = ac_falsy.aggregated();
                    let ca = s.list().unwrap();
                    check_length(ca, mask)?;
                    let mut out: ListChunked = ca
                        .into_iter()
                        .zip(mask.into_iter())
                        .map(|(falsy, take)| match (falsy, take) {
                            (Some(_), Some(true)) => None,
                            (Some(v), Some(false)) => Some(v),
                            _ => None,
                        })
                        .collect_trusted();
                    out.rename(ac_truthy.series().name());
                    ac_truthy.with_series(out.into_series(), true);
                    Ok(ac_truthy)
                }
                // then:
                //     col().shift()
                // otherwise:
                //     lit(list)
                else if ac_truthy.is_literal() {
                    let literal = ac_truthy.series();
                    let s = ac_falsy.aggregated();
                    let ca = s.list().unwrap();
                    check_length(ca, mask)?;
                    let mut out: ListChunked = ca
                        .into_iter()
                        .zip(mask.into_iter())
                        .map(|(falsy, take)| match (falsy, take) {
                            (Some(_), Some(true)) => Some(literal.clone()),
                            (Some(v), Some(false)) => Some(v),
                            _ => None,
                        })
                        .collect_trusted();
                    out.rename(ac_truthy.series().name());
                    ac_truthy.with_series(out.into_series(), true);
                    Ok(ac_truthy)
                } else {
                    let literal = ac_falsy.series();
                    let s = ac_truthy.aggregated();
                    let ca = s.list().unwrap();
                    check_length(ca, mask)?;
                    let mut out: ListChunked = ca
                        .into_iter()
                        .zip(mask.into_iter())
                        .map(|(truthy, take)| match (truthy, take) {
                            (Some(v), Some(true)) => Some(v),
                            (Some(_), Some(false)) => Some(literal.clone()),
                            _ => None,
                        })
                        .collect_trusted();
                    out.rename(ac_truthy.series().name());
                    ac_truthy.with_series(out.into_series(), true);
                    Ok(ac_truthy)
                }
            }
            // Both are or a flat series or aggregated into a list
            // so we can flatten the Series an apply the operators
            _ => {
                // inspect the predicate and if it is consisting
                // if arity/binary and some aggreation we apply as iters as
                // it gets complicated quickly.
                // For instance:
                //  when(col(..) > min(..)).then(..).otherwise(..)
                if let Some(expr) = self.predicate.as_expression() {
                    let mut has_arity = false;
                    let mut has_agg = false;
                    for e in expr.into_iter() {
                        match e {
                            Expr::BinaryExpr { .. } | Expr::Ternary { .. } => has_arity = true,
                            Expr::Agg(_) => has_agg = true,
                            Expr::Function { options, .. }
                            | Expr::AnonymousFunction { options, .. }
                                if options.is_groups_sensitive() =>
                            {
                                has_agg = true
                            }
                            _ => {}
                        }
                    }
                    if has_arity && has_agg {
                        return finish_as_iters(ac_truthy, ac_falsy, ac_mask);
                    }
                }

                if !aggregation_predicate {
                    // experimental elementwise behavior tested in `test_binary_agg_context_1`
                    return finish_as_iters(ac_truthy, ac_falsy, ac_mask);
                }
                let mut mask = mask_s.bool()?.clone();
                let mut truthy = ac_truthy.flat_naive().into_owned();
                let mut falsy = ac_falsy.flat_naive().into_owned();
                expand_lengths(&mut truthy, &mut falsy, &mut mask);
                let out = truthy.zip_with(&mask, &falsy)?;

                // because of the flattening we don't have to do that anymore
                if matches!(ac_truthy.update_groups, UpdateGroups::WithSeriesLen) {
                    ac_truthy.with_update_groups(UpdateGroups::No);
                }

                ac_truthy.with_series(out, false);

                Ok(ac_truthy)
            }
        }
    }
    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }

    fn is_valid_aggregation(&self) -> bool {
        self.truthy.is_valid_aggregation() || self.falsy.is_valid_aggregation()
    }
}

impl PartitionedAggregation for TernaryExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        let truthy = self.truthy.as_partitioned_aggregator().unwrap();
        let falsy = self.falsy.as_partitioned_aggregator().unwrap();
        let mask = self.predicate.as_partitioned_aggregator().unwrap();

        let mut truthy = truthy.evaluate_partitioned(df, groups, state)?;
        let mut falsy = falsy.evaluate_partitioned(df, groups, state)?;
        let mask = mask.evaluate_partitioned(df, groups, state)?;
        let mut mask = mask.bool()?.clone();

        expand_lengths(&mut truthy, &mut falsy, &mut mask);
        truthy.zip_with(&mask, &falsy)
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
