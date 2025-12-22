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
    allow_threading: bool,
}

impl StructEvalExpr {
    pub(crate) fn new(
        input: Arc<dyn PhysicalExpr>,
        evaluation: Vec<Arc<dyn PhysicalExpr>>,
        expr: Expr,
        output_field: Field,
        allow_threading: bool,
    ) -> Self {
        Self {
            input,
            evaluation,
            expr,
            output_field,
            allow_threading,
        }
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

        state.with_fields = Arc::new(Some(input.struct_()?.clone()));

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
        };
        for col in cols? {
            eval.push(col);
        }

        // Apply with_fields.
        with_fields(&eval)
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        // Evaluate input.
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;
        ac.set_groups_for_undefined_agg_states();

        // Set ExecutionState.
        let mut state = state.clone();
        state.with_fields_ac = Arc::new(Some(ac.into_static()));

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
        };
        for ac in acs_eval? {
            acs.push(ac)
        }

        // Apply with_fields. We default to group_aware as groups may have diverged.
        // @TODO. Performance can be optimized: elementwise, re-use PlIndexMap, parallel evaluation, etc.
        let mut iters = acs
            .iter_mut()
            .map(|ac| ac.iter_groups(true))
            .collect::<Vec<_>>();
        let len = iters[0].size_hint().0;
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

        // Update AC, same logic as ApplyExpr.
        let mut ac = acs.swap_remove(0);
        let col = if matches!(
            ac.agg_state(),
            AggState::AggregatedScalar(_) | AggState::LiteralScalar(_)
        ) {
            let out = ca.explode(ExplodeOptions {
                empty_as_null: true,
                keep_nulls: true,
            })?;
            ac.update_groups = UpdateGroups::No;
            out.into_column()
        } else {
            ac.with_update_groups(UpdateGroups::WithSeriesLen);
            ca.into_series().into()
        };

        ac.with_values_and_args(col, true, self.as_expression(), false, self.is_scalar())?;
        Ok(ac)
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn is_scalar(&self) -> bool {
        false
    }
}
