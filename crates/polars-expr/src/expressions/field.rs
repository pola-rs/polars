use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, Field, GroupPositions};
use polars_plan::dsl::Expr;

use super::*;
use crate::prelude::AggregationContext;
use crate::state::ExecutionState;

#[derive(Clone)]
pub struct FieldExpr {
    name: PlSmallStr,
    expr: Expr,
    output_field: Field,
}

impl FieldExpr {
    pub fn new(name: PlSmallStr, expr: Expr, output_field: Field) -> Self {
        Self {
            name,
            expr,
            output_field,
        }
    }
}

impl PhysicalExpr for FieldExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, _df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let (ca, _validity) = state.with_fields.as_ref().clone().ok_or_else(
            || polars_err!(invalid_field_use),
        )?;

        ca.field_by_name(self.name.as_str()).map(Column::from)
    }

    fn evaluate_on_groups<'a>(
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
