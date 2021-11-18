use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use std::sync::Arc;

pub struct CastExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) data_type: DataType,
    pub(crate) expr: Expr,
    pub(crate) strict: bool,
}

impl CastExpr {
    fn finish(&self, input: &Series) -> Result<Series> {
        // this is quite dirty
        // We use the booleanarray as null series, because we have no null array.
        // in a ternary or binary operation, we then do type coercion to matching supertype.
        // here we create a null array for the types we cannot cast to from a booleanarray

        if input.bool().is_ok() && input.null_count() == input.len() {
            match self.data_type {
                DataType::List(_) => {
                    return Ok(ListChunked::full_null(input.name(), input.len()).into_series())
                }
                #[cfg(feature = "dtype-date")]
                DataType::Date => {
                    return Ok(Int32Chunked::full_null(input.name(), input.len())
                        .into_date()
                        .into_series())
                }
                #[cfg(feature = "dtype-datetime")]
                DataType::Datetime => {
                    return Ok(Int64Chunked::full_null(input.name(), input.len())
                        .into_date()
                        .into_series())
                }
                _ => {}
            }
        }

        if self.strict {
            input.strict_cast(&self.data_type)
        } else {
            input.cast(&self.data_type)
        }
    }
}

impl PhysicalExpr for CastExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.input.evaluate(df, state)?;
        self.finish(&series)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;
        let s = ac.flat();
        let s = self.finish(s.as_ref())?;
        ac.with_series(s, false);
        Ok(ac)
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}
