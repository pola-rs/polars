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

    // In-memory engine only.
    fn evaluate_impl(&self, _df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let ca = state
            .with_fields
            .as_ref()
            .ok_or_else(|| polars_err!(invalid_field_use))?;

        ca.field_by_name(self.name.as_str()).map(Column::from)
    }

    // In-memory engine only.
    fn evaluate_on_groups_impl<'a>(
        &self,
        _df: &DataFrame,
        _groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let ac = state
            .with_fields_ac
            .as_ref()
            .ok_or_else(|| polars_err!(invalid_field_use))?;

        let name = self.name.as_str();

        let state = match ac.agg_state() {
            AggState::AggregatedList(c) => {
                let out = c
                    .list()?
                    .apply_to_inner(&|s| s.struct_()?.field_by_name(name))?;
                AggState::AggregatedList(out.into_column())
            },
            AggState::NotAggregated(c) => {
                AggState::NotAggregated(c.struct_()?.field_by_name(name)?.into_column())
            },
            AggState::AggregatedScalar(c) => {
                AggState::AggregatedScalar(c.struct_()?.field_by_name(name)?.into_column())
            },
            AggState::LiteralScalar(c) => {
                AggState::LiteralScalar(c.struct_()?.field_by_name(name)?.into_column())
            },
        };

        Ok(AggregationContext {
            state,
            groups: ac.groups.clone(),
            update_groups: ac.update_groups,
            original_groups: ac.original_groups,
        })
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn is_scalar(&self) -> bool {
        false
    }
}
