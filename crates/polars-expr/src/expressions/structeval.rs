use std::sync::Arc;

use polars_core::POOL;
use polars_core::error::{PolarsResult, polars_ensure};
use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_core::schema::Schema;
use polars_plan::dsl::Expr;
use rayon::prelude::*;

use super::PhysicalExpr;
#[cfg(feature = "dtype-struct")]
use crate::dispatch::struct_::with_fields;
use crate::prelude::{AggState, AggregationContext, UpdateGroups};
use crate::state::ExecutionState;

#[derive(Clone)]
pub struct StructEvalExpr {
    input: Arc<dyn PhysicalExpr>,
    evaluation: Vec<Arc<dyn PhysicalExpr>>,
    expr: Expr,
    output_field: Field,
    operates_on_scalar: bool,
    allow_threading: bool,
}

impl StructEvalExpr {
    pub(crate) fn new(
        input: Arc<dyn PhysicalExpr>,
        evaluation: Vec<Arc<dyn PhysicalExpr>>,
        expr: Expr,
        output_field: Field,
        operates_on_scalar: bool,
        allow_threading: bool,
    ) -> Self {
        Self {
            input,
            evaluation,
            expr,
            output_field,
            operates_on_scalar,
            allow_threading,
        }
    }
}

impl StructEvalExpr {
    fn apply_all_literal_elementwise<'a>(
        &self,
        mut acs: Vec<AggregationContext<'a>>,
    ) -> PolarsResult<AggregationContext<'a>> {
        let cols = acs
            .iter()
            .map(|ac| ac.get_values().clone())
            .collect::<Vec<_>>();
        let out = with_fields(&cols)?;
        polars_ensure!(
            out.len() == 1,
            ComputeError: "elementwise expression {:?} must return exactly 1 value on literals, got {}",
                &self.expr, out.len()
        );
        let mut ac = acs.pop().unwrap();
        ac.with_literal(out);
        Ok(ac)
    }

    fn apply_elementwise<'a>(
        &self,
        mut acs: Vec<AggregationContext<'a>>,
        must_aggregate: bool,
    ) -> PolarsResult<AggregationContext<'a>> {
        // At this stage, we either have (with or without LiteralScalars):
        // - one or more AggregatedList or NotAggregated ACs
        // - one or more AggregatedScalar ACs

        let mut previous = None;
        for ac in acs.iter_mut() {
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

        // The first AC represents the `input` and will be used as the base AC.
        let base_ac_idx = 0;

        match acs[base_ac_idx].agg_state() {
            AggState::AggregatedList(s) => {
                let aggregated = acs.iter().any(|ac| ac.is_aggregated());
                let ca = s.list().unwrap();
                let input_len = s.len();

                let out = ca.apply_to_inner(&|_| {
                    let cols = acs
                        .iter()
                        .map(|ac| ac.flat_naive().into_owned())
                        .collect::<Vec<_>>();
                    Ok(with_fields(&cols)?.as_materialized_series().clone())
                })?;

                let out = out.into_column();
                assert!(input_len == out.len());

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
                assert!(aggregated == self.is_scalar());

                let cols = acs
                    .iter()
                    .map(|ac| ac.flat_naive().into_owned())
                    .collect::<Vec<_>>();

                let input_len = cols[base_ac_idx].len();
                let out = with_fields(&cols)?;
                assert!(input_len == out.len());

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

    fn apply_group_aware<'a>(
        &self,
        mut acs: Vec<AggregationContext<'a>>,
    ) -> PolarsResult<AggregationContext<'a>> {
        let len = acs[0].groups.len();
        let mut iters = acs
            .iter_mut()
            .map(|ac| ac.iter_groups(true))
            .collect::<Vec<_>>();
        let ca = (0..len)
            .map(|_| {
                let mut cols = Vec::with_capacity(iters.len());
                for i in &mut iters {
                    match i.next().unwrap() {
                        None => return Ok(None),
                        Some(s) => cols.push(s.as_ref().clone().into_column()),
                    }
                }
                let out = with_fields(&cols)?;
                Ok(Some(out))
            })
            .collect::<PolarsResult<ListChunked>>()?;
        drop(iters);

        // Finish apply groups; see also ApplyExpr for the reference solution.
        let ac = acs.swap_remove(0);
        self.finish_apply_groups(ac, ca)
    }

    fn finish_apply_groups<'a>(
        &self,
        mut ac: AggregationContext<'a>,
        ca: ListChunked,
    ) -> PolarsResult<AggregationContext<'a>> {
        let col = if self.is_scalar() {
            let out = ca
                .explode(ExplodeOptions {
                    empty_as_null: true,
                    keep_nulls: true,
                })
                .unwrap();
            // if the explode doesn't return the same len, it wasn't scalar.
            polars_ensure!(out.len() == ca.len(), InvalidOperation: "expected scalar for expr: {}, got {}", self.expr, &out);
            ac.update_groups = UpdateGroups::No;
            out.into_column()
        } else {
            ac.with_update_groups(UpdateGroups::WithSeriesLen);
            ca.into_series().into()
        };

        ac.with_values_and_args(col, true, self.as_expression(), false, self.is_scalar())?;

        Ok(ac)
    }
}

impl PhysicalExpr for StructEvalExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let input = self.input.evaluate(df, state)?;

        // Set ExecutionState.
        let mut state = state.clone();
        let mut eval = Vec::with_capacity(self.evaluation.len() + 1);
        let input_len = input.len();

        state.with_fields = Some(Arc::new(input.struct_()?.clone()));

        // Collect evaluation fields; input goes first.
        eval.push(input);

        let f = |e: &Arc<dyn PhysicalExpr>| {
            let result = e.evaluate(df, &state)?;
            polars_ensure!(
                result.len() == input_len || result.len() == 1,
                ShapeMismatch: "struct.with_fields expressions must have matching or unit length"
            );
            Ok(result)
        };
        let cols = if self.allow_threading {
            POOL.install(|| {
                self.evaluation
                    .par_iter()
                    .map(f)
                    .collect::<PolarsResult<Vec<_>>>()
            })
        } else {
            self.evaluation
                .iter()
                .map(f)
                .collect::<PolarsResult<Vec<_>>>()
        }?;
        eval.extend(cols);

        // Apply with_fields.
        with_fields(&eval)
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        // The evaluation is similar to a regular Function, with the modification that the input
        // is evaluated first, and retained for future use in the ExecutionState.

        // Evaluate input.
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;

        ac.groups();
        ac.set_groups_for_undefined_agg_states();

        // Snap the AC into the ExecutionState for re-use when Field is evaluated.
        let mut state = state.clone();
        state.with_fields_ac = Some(Arc::new(ac.into_static()));

        // Collect evaluation fields.
        let mut acs = Vec::with_capacity(self.evaluation.len() + 1);
        acs.push(ac);

        let f = |e: &Arc<dyn PhysicalExpr>| e.evaluate_on_groups(df, groups, &state);
        let acs_eval = if self.allow_threading {
            POOL.install(|| {
                self.evaluation
                    .par_iter()
                    .map(f)
                    .collect::<PolarsResult<Vec<_>>>()
            })
        } else {
            self.evaluation
                .iter()
                .map(f)
                .collect::<PolarsResult<Vec<_>>>()
        }?;
        acs.extend(acs_eval);

        // Revert ExecutionState.
        state.with_fields_ac = None;

        // Merge the `evaluation` back into the `input` struct.
        // @NOTE. From this point on, we are dealing with a regular Function `with_fields`, which is
        // elementwise top-level and not fallible. We leverage the reference dispatch for ApplyExpr,
        // but simplified.

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
                        not_agg_groups_may_diverge |= !p.groups.is_same(&ac.groups)
                    }
                    previous = Some(ac);
                    if ac.groups.is_overlapping() {
                        has_not_agg_with_overlapping_groups = true;
                    }
                },
                AggState::LiteralScalar(_) => {},
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
            self.apply_group_aware(acs)
        } else if elementwise_must_aggregate && has_not_agg_with_overlapping_groups {
            // Compatible but calling aggregated() is too expensive
            self.apply_group_aware(acs)
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
                            acs[base_ac_idx]
                                .groups
                                .iter()
                                .zip(ac.groups.iter())
                                .any(|(l, r)| l.len() != r.len() && (l.len() == 1 || r.len() == 1))
                        })
                } else {
                    false
                };
            if has_broadcast {
                //  Broadcast fall-back.
                self.apply_group_aware(acs)
            } else {
                self.apply_elementwise(acs, elementwise_must_aggregate)
            }
        }
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn is_scalar(&self) -> bool {
        self.operates_on_scalar
    }
}
