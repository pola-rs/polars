use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;

pub struct ApplyExpr {
    pub inputs: Vec<Arc<dyn PhysicalExpr>>,
    pub function: NoEq<Arc<dyn SeriesUdf>>,
    pub output_type: Option<DataType>,
    pub expr: Expr,
    pub collect_groups: ApplyOptions,
}

impl PhysicalExpr for ApplyExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let mut inputs = self
            .inputs
            .par_iter()
            .map(|e| e.evaluate(df, state))
            .collect::<Result<Vec<_>>>()?;
        let in_name = inputs[0].name().to_string();
        let mut out = self.function.call_udf(&mut inputs)?;
        if in_name != out.name() {
            out.rename(&in_name);
        }
        Ok(out)
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        if self.inputs.len() > 1 {
            return Err(PolarsError::InvalidOperation(
                "function with multiple inputs not yet supported in aggregation context".into(),
            ));
        }
        let mut ac = if let Ok(ae) = self.inputs[0].as_agg_expr() {
            AggregationContext::new(
                ae.aggregate(df, groups, state)?.unwrap(),
                Cow::Borrowed(groups),
            )
        } else {
            self.inputs[0].evaluate_on_groups(df, groups, state)?
        };

        match self.collect_groups {
            ApplyOptions::ApplyGroups => {
                let mut container = vec![Default::default()];
                let name = ac.series().name().to_string();

                let mut ca: ListChunked = ac
                    .aggregated()
                    .list()
                    .unwrap()
                    .into_iter()
                    .map(|opt_s| {
                        opt_s.and_then(|s| {
                            container[0] = s;
                            self.function.call_udf(&mut container).ok()
                        })
                    })
                    .collect();
                ca.rename(&name);
                ac.with_series(ca.into_series());
                Ok(ac)
            }
            ApplyOptions::ApplyFlat => {
                let s = self.function.call_udf(&mut [ac.take()])?;
                ac.with_series(s);
                Ok(ac)
            }
            ApplyOptions::ApplyList => {
                let s = self
                    .function
                    .call_udf(&mut [ac.aggregated().into_owned()])?;
                ac.with_series(s);
                Ok(ac)
            }
        }
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
        if self.inputs.len() > 1 {
            return Err(PolarsError::InvalidOperation(
                "function with multiple inputs not yet supported in aggregation context".into(),
            ));
        }
        let mut ac = self.inputs[0].evaluate_on_groups(df, groups, state)?;

        match self.collect_groups {
            ApplyOptions::ApplyGroups => {
                let mut container = vec![Default::default()];
                let name = ac.series().name().to_string();

                let mut all_unit_length = true;

                let mut ca: ListChunked = ac
                    .aggregated()
                    .list()
                    .unwrap()
                    .into_iter()
                    .map(|opt_s| {
                        opt_s.and_then(|s| {
                            container[0] = s;
                            let out = self.function.call_udf(&mut container).ok();

                            if let Some(s) = &out {
                                if s.len() != 1 {
                                    all_unit_length = false;
                                }
                            }
                            out
                        })
                    })
                    .collect();
                ca.rename(&name);
                if all_unit_length {
                    return Ok(Some(ca.explode()?));
                }
                Ok(Some(ca.into_series()))
            }
            ApplyOptions::ApplyFlat => self.function.call_udf(&mut [ac.take()]).map(Some),
            ApplyOptions::ApplyList => self
                .function
                .call_udf(&mut [ac.aggregated().into_owned()])
                .map(Some),
        }
    }
}
