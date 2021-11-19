use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;

pub struct ColumnExpr(Arc<str>, Expr);

impl ColumnExpr {
    pub fn new(name: Arc<str>, expr: Expr) -> Self {
        Self(name, expr)
    }
}

impl PhysicalExpr for ColumnExpr {
    fn as_expression(&self) -> &Expr {
        &self.1
    }
    fn evaluate(&self, df: &DataFrame, _state: &ExecutionState) -> Result<Series> {
        let column = match &*self.0 {
            "" => df.select_at_idx(0).ok_or_else(|| {
                PolarsError::NoData("could not select a column from an empty DataFrame".into())
            })?,
            _ => df.column(&self.0)?,
        };
        Ok(column.clone())
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let s = self.evaluate(df, state)?;
        Ok(AggregationContext::new(s, Cow::Borrowed(groups), false))
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field = input_schema.field_with_name(&self.0).map(|f| f.clone())?;
        Ok(field)
    }
}
