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
        let ca = state
            .with_fields
            .as_ref()
            .as_ref()
            .ok_or_else(|| polars_err!(invalid_field_use))?;

        ca.field_by_name(self.name.as_str()).map(Column::from)
    }

    fn evaluate_on_groups<'a>(
        &self,
        _df: &DataFrame,
        _groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        //kdn TODO REVIEW
        let ac = state
            .with_fields_ac
            .as_ref()
            .as_ref()
            .ok_or_else(|| polars_err!(invalid_field_use))?;

        let col = ac.flat_naive().clone();
        let ca = col.struct_()?;
        let out = ca.field_by_name(self.name.as_str()).map(Column::from)?;

        Ok(AggregationContext::new(out, ac.groups.clone(), false))
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn is_scalar(&self) -> bool {
        false
    }
}
