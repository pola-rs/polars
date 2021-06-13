use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;
use std::sync::Arc;

pub struct CastExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) data_type: DataType,
}

impl CastExpr {
    pub fn new(input: Arc<dyn PhysicalExpr>, data_type: DataType) -> Self {
        Self { input, data_type }
    }
}

impl PhysicalExpr for CastExpr {
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.input.evaluate(df, state)?;
        // this is quite dirty
        // We use the booleanarray as null series, because we have no null array.
        // in a ternary or binary operation, we then do type coercion to matching supertype.
        // here we create a null array for the types we cannot cast to from a booleanarray
        if matches!(self.data_type, DataType::List(_)) {
            // the booleanarray is hacked as null type
            if series.bool().is_ok() && series.null_count() == series.len() {
                return Ok(ListChunked::full_null(series.name(), series.len()).into_series());
            }
        }
        series.cast_with_dtype(&self.data_type)
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}
