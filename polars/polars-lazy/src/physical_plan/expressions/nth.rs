use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_arrow::index::IndexToUsize;
use polars_core::prelude::*;
use std::borrow::Cow;

pub struct NthExpr {
    expr: Expr,
}

impl NthExpr {
    pub(crate) fn new(i: i64) -> Self {
        Self { expr: Expr::Nth(i) }
    }
}

impl PhysicalExpr for NthExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, _state: &ExecutionState) -> Result<Series> {
        let idx = if let Expr::Nth(i) = self.expr {
            i
        } else {
            unreachable!()
        };
        let idx = idx
            .negative_to_usize(df.width())
            .ok_or_else(|| PolarsError::NoData("cannot take nth from empty dataframe".into()))?;
        Ok(df.get_columns()[idx].clone())
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let s = self.evaluate(df, state)?;
        Ok(AggregationContext::new(s, Cow::Borrowed(groups), false))
    }
    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        Ok(Field::new("count", DataType::UInt32))
    }
}

impl PhysicalAggregation for NthExpr {
    fn aggregate(
        &self,
        _df: &DataFrame,
        _groups: &GroupsProxy,
        _state: &ExecutionState,
    ) -> Result<Option<Series>> {
        Ok(None)
    }
}
