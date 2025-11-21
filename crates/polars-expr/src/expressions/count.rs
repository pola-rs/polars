use std::borrow::Cow;

use polars_core::prelude::*;
use polars_plan::constants::LEN;

use super::*;
use crate::expressions::{AggregationContext, PhysicalExpr};

pub struct CountExpr {
    expr: Expr,
}

impl CountExpr {
    pub(crate) fn new() -> Self {
        Self { expr: Expr::Len }
    }
}

impl PhysicalExpr for CountExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, _state: &ExecutionState) -> PolarsResult<Column> {
        Ok(Column::new_scalar(
            PlSmallStr::from_static(LEN),
            Scalar::from(df.height() as IdxSize),
            1,
        ))
    }

    fn evaluate_on_groups<'a>(
        &self,
        _df: &DataFrame,
        groups: &'a GroupPositions,
        _state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let ca = groups.group_count().with_name(PlSmallStr::from_static(LEN));
        let c = ca.into_column();
        Ok(AggregationContext::new(c, Cow::Borrowed(groups), true))
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(Field::new(PlSmallStr::from_static(LEN), IDX_DTYPE))
    }

    fn is_scalar(&self) -> bool {
        true
    }
}
