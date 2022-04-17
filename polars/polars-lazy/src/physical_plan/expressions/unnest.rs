use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;

pub struct UnnestExpr(Arc<str>, Expr);

impl UnnestExpr {
    pub fn new(path: Arc<str>, expr: Expr) -> Self {
        Self(path, expr)
    }
}

impl PhysicalExpr for UnnestExpr {
    fn as_expression(&self) -> &Expr {
        &self.1
    }
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        // TODO: Add real operation :)
        // TODO: Refactor to DataFrame.nested_column or something similar
        let columns: Vec<&str> = self.0.split('.').collect();
        let mut series = df.column(columns[0]).map(|s| s.clone())?;

        for column in columns[1..].iter() {
            let struct_array = series.struct_()?;
            series = struct_array.field_by_name(column)?;
        }

        // Use the full nested name
        series.rename(&self.0);

        Ok(series)
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let s = self.evaluate(df, state)?;
        Ok(AggregationContext::new(s, Cow::Borrowed(groups), false))
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field = input_schema.get_field(&self.0).ok_or_else(|| {
            PolarsError::NotFound(format!(
                "could not find column: {} in schema: {:?}",
                self.0, &input_schema
            ))
        })?;
        Ok(field)
    }
}
