use std::sync::Arc;

use polars_core::frame::group_by::GroupsProxy;
use polars_core::prelude::*;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct InnerStructExpr {
    input: Arc<dyn PhysicalExpr>,
    struct_exprs: Vec<Expr>,
    expr: Expr,
}

impl InnerStructExpr {
    pub(crate) fn new(input: Arc<dyn PhysicalExpr>, struct_exprs: Vec<Expr>, expr: Expr) -> Self {
        Self {
            input,
            struct_exprs,
            expr,
        }
    }
}

impl PhysicalExpr for InnerStructExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let input = self.input.evaluate(df, state)?;

        let name = input.name();
        let inner_df = input.struct_()?.clone().unnest();

        Ok(inner_df
            .lazy()
            .select(&self.struct_exprs)
            .collect()?
            .into_struct(&name)
            .into_series())
    }

    fn evaluate_on_groups<'a>(
        &self,
        _df: &DataFrame,
        _groups: &'a GroupsProxy,
        _state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        todo!()
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        todo!()
    }
}
