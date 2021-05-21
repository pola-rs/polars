use crate::logical_plan::Context;
use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::{prelude::*, POOL};
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
impl PhysicalAggregation for BinaryFunctionExpr {
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let (agg_a, agg_b): (Result<Series>, Result<Series>) = POOL.install(|| {
            rayon::join(
                || {
                    let a = self.input_a.evaluate(df, state)?;
                    Ok(a.agg_list(groups).expect("no data"))
                },
                || {
                    let b = self.input_b.evaluate(df, state)?;
                    Ok(b.agg_list(groups).expect("no data"))
                },
            )
        });

        // keep track of the output lengths. If they are all unit length,
        // we can explode the array as it would have the same length as the no. of groups
        // if it is not all unit length it should remain a listarray

        let mut all_unit_length = true;

        let ca = agg_a?
            .list()
            .unwrap()
            .into_iter()
            .zip(agg_b?.list().unwrap())
            .map(|(opt_a, opt_b)| match (opt_a, opt_b) {
                (Some(a), Some(b)) => {
                    let out = self.function.call_udf(a, b).ok();

                    if let Some(s) = &out {
                        if s.len() != 1 {
                            all_unit_length = false;
                        }
                    }
                    out
                }
                _ => None,
            })
            .collect::<ListChunked>();

        if all_unit_length {
            return Ok(Some(ca.explode()?));
        }
        Ok(Some(ca.into_series()))
    }
}
