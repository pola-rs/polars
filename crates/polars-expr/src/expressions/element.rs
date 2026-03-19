use polars_core::prelude::*;

use super::*;
use crate::expressions::{AggregationContext, PhysicalExpr};

#[derive(Debug)]
pub struct ElementExpr {
    output_field: Field,
}

impl ElementExpr {
    pub fn new(output_field: Field) -> Self {
        Self { output_field }
    }
}

impl PhysicalExpr for ElementExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&Expr::Element)
    }

    fn evaluate_impl(&self, _df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let (flattened, _validity) = state.element.as_ref().clone().ok_or_else(
            || polars_err!(InvalidOperation: "`element` is not allowed in this context"),
        )?;
        Ok(flattened)
    }

    fn evaluate_on_groups_impl<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let c = self.evaluate(df, state)?;
        Ok(AggregationContext::new(c, Cow::Borrowed(groups), false))
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn is_scalar(&self) -> bool {
        false
    }
}
