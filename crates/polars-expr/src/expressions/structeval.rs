use std::borrow::Cow;
use std::sync::Arc;

use polars_core::error::{PolarsResult, polars_ensure};
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, Field, GroupPositions};
use polars_core::schema::Schema;
use polars_plan::dsl::Expr;

use super::PhysicalExpr;
use crate::dispatch::struct_::with_fields;
use crate::prelude::AggregationContext;
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
        let validity = input.rechunk_validity();

        // Set ExecutionState.
        let mut state = state.clone();
        let mut eval = Vec::with_capacity(self.evaluation.len() + 1);
        let input_len = input.len();

        state.with_fields = Arc::new(Some(input.struct_()?.clone()));

        // Collect evaluation fields.
        eval.push(input);
        for e in self.evaluation.iter() {
            let result = e.evaluate(&df, &state)?;
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
        //kdn: TODO TEST
        // - elementwise: sum
        // - not row-separable: cum_sum
        // - aggregations: first
        // - window..

        // Evaluate input.
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;
        let input = ac.get_values().clone();
        let validity = input.rechunk_validity();
        dbg!(&ac);

        // Set ExecutionState.
        let mut state = state.clone();
        let mut eval = Vec::with_capacity(self.evaluation.len() + 1);
        let input_len = input.len();

        state.with_fields = Arc::new(Some(input.struct_()?.clone()));

        // TBD which groups?

        // Collect evaluation fields.
        eval.push(input);
        for e in self.evaluation.iter() {
            let result = e.evaluate_on_groups(&df, ac.groups(), &state)?;
            dbg!(&result);
            //kdn TODO length check
            // polars_ensure!(result.len() == input_len || result.len() == 1,
            //     ShapeMismatch: "struct.with_fields expressions must have matching or unit length"
            // );
            let result = result.flat_naive();
            dbg!(&result);
            eval.push(result.into_owned());
        }

        // Apply with_fields.
        ac.with_values(with_fields(&eval)?, false, Some(&self.expr))?;
        Ok(ac)
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn is_scalar(&self) -> bool {
        false
    }
}
