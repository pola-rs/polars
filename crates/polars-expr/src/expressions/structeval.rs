use std::sync::Arc;

use polars_core::error::{PolarsResult, polars_ensure};
use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_core::schema::Schema;
use polars_plan::dsl::Expr;

use super::PhysicalExpr;
use crate::dispatch::struct_::with_fields;
use crate::prelude::{AggState, AggregationContext, UpdateGroups};
use crate::state::ExecutionState;

#[derive(Clone)]
pub struct StructEvalExpr {
    input: Arc<dyn PhysicalExpr>,
    evaluation: Vec<Arc<dyn PhysicalExpr>>,
    expr: Expr,
    output_field: Field,
}

impl StructEvalExpr {
    pub(crate) fn new(
        input: Arc<dyn PhysicalExpr>,
        evaluation: Vec<Arc<dyn PhysicalExpr>>,
        expr: Expr,
        output_field: Field,
    ) -> Self {
        Self {
            input,
            evaluation,
            expr,
            output_field,
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

        // Collect evaluation fields.
        eval.push(input);
        for e in self.evaluation.iter() {
            let result = e.evaluate(df, &state)?;
            polars_ensure!(result.len() == input_len || result.len() == 1,
                ShapeMismatch: "struct.with_fields expressions must have matching or unit length"
            );
            eval.push(result);
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
        ac.groups();

        // Set ExecutionState.
        let mut state = state.clone();
        state.with_fields_ac = Arc::new(Some(ac.into_static()));

        // Collect evaluation fields.
        let mut acs = Vec::with_capacity(self.evaluation.len() + 1);
        acs.push(ac);
        for e in self.evaluation.iter() {
            let ac = e.evaluate_on_groups(df, groups, &state)?;
            acs.push(ac);
        }

        // Apply with_fields.
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

        // Update AC.
        let mut ac = acs.swap_remove(0);
        ac.with_agg_state(AggState::AggregatedList(ca.into_column()));
        ac.with_update_groups(UpdateGroups::WithSeriesLen);
        Ok(ac)
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn is_scalar(&self) -> bool {
        false
    }
}
