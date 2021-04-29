use crate::logical_plan::Context;
use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_core::prelude::*;
use std::sync::Arc;

pub(crate) struct BinaryFunctionExpr {
    pub(crate) input_a: Arc<dyn PhysicalExpr>,
    pub(crate) input_b: Arc<dyn PhysicalExpr>,
    pub(crate) function: NoEq<Arc<dyn SeriesBinaryUdf>>,
    pub(crate) output_field: NoEq<Arc<dyn BinaryUdfOutputField>>,
}

impl PhysicalExpr for BinaryFunctionExpr {
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series_a = self.input_a.evaluate(df, state)?;
        let series_b = self.input_b.evaluate(df, state)?;

        self.function.call_udf(series_a, series_b).map(|mut s| {
            s.rename("binary_function");
            s
        })
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field_a = self.input_a.to_field(input_schema)?;
        let field_b = self.input_b.to_field(input_schema)?;
        self.output_field
            .get_field(input_schema, Context::Default, &field_a, &field_b)
            .ok_or_else(|| PolarsError::UnknownSchema("no field found".into()))
    }
    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}
