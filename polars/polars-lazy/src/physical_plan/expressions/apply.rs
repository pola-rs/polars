use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

pub struct ApplyExpr {
    pub inputs: Vec<Arc<dyn PhysicalExpr>>,
    pub function: NoEq<Arc<dyn SeriesUdf>>,
    pub output_type: Option<DataType>,
    pub expr: Expr,
}

impl ApplyExpr {
    pub fn new(
        input: Vec<Arc<dyn PhysicalExpr>>,
        function: NoEq<Arc<dyn SeriesUdf>>,
        output_type: Option<DataType>,
        expr: Expr,
    ) -> Self {
        ApplyExpr {
            inputs: input,
            function,
            output_type,
            expr,
        }
    }
}

impl PhysicalExpr for ApplyExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let inputs = self
            .inputs
            .iter()
            .map(|e| e.evaluate(df, state))
            .collect::<Result<Vec<_>>>()?;
        let in_name = inputs[0].name().to_string();
        let mut out = self.function.call_udf(inputs)?;
        if in_name != out.name() {
            out.rename(&in_name);
        }
        Ok(out)
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        match &self.output_type {
            Some(output_type) => {
                let input_field = self.inputs[0].to_field(input_schema)?;
                Ok(Field::new(input_field.name(), output_type.clone()))
            }
            None => self.inputs[0].to_field(input_schema),
        }
    }
    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

impl PhysicalAggregation for ApplyExpr {
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        // two possible paths
        // all inputs may be final aggregations
        // or they may be expression that can work on groups but not yet produce an aggregation

        // we first collect the inputs
        // if any of the input aggregations yields None, we return None as well
        // we check this by comparing the length of the inputs before and after aggregation
        let inputs: Vec<_> = match self.inputs[0].as_agg_expr() {
            Ok(_) => {
                let inputs = self
                    .inputs
                    .par_iter()
                    .map(|e| {
                        let e = e.as_agg_expr()?;
                        e.aggregate(df, groups, state)
                    })
                    .collect::<Result<Vec<_>>>()?;
                inputs.into_iter().flatten().collect()
            }
            _ => {
                let inputs = self
                    .inputs
                    .par_iter()
                    .map(|e| {
                        let (s, groups) = e.evaluate_on_groups(df, groups, state)?;
                        Ok(s.agg_list(&groups))
                    })
                    .collect::<Result<Vec<_>>>()?;
                inputs.into_iter().flatten().collect()
            }
        };

        if inputs.len() == self.inputs.len() {
            self.function.call_udf(inputs).map(Some)
        } else {
            Ok(None)
        }
    }
}
