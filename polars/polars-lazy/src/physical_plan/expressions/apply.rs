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
    pub collect_groups: bool,
}

impl PhysicalExpr for ApplyExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let mut inputs = self
            .inputs
            .iter()
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
    ) -> Result<(Series, Cow<'a, GroupTuples>)> {
        let mut owned_count = 0;
        let mut inputs = Vec::with_capacity(self.inputs.len());
        let mut groups_vec = Vec::with_capacity(self.inputs.len());
        let mut owned_group = None;

        self.inputs.iter().try_for_each::<_, Result<_>>(|e| {
            let (s, groups_) = e.evaluate_on_groups(df, groups, state)?;
            inputs.push(s);
            if let Cow::Owned(_) = &groups_ {
                owned_group = Some(groups_);
                owned_count += 1;
                return Ok(());
            }
            groups_vec.push(groups_);
            Ok(())
        })?;

        let in_name = inputs[0].name().to_string();
        let mut out = self.function.call_udf(&mut inputs)?;
        if in_name != out.name() {
            out.rename(&in_name);
        }

        match owned_count {
            0 => Ok((out, groups_vec.pop().unwrap())),
            1 => Ok((out, owned_group.unwrap())),
            _ => Err(PolarsError::ValueError(
                "Function may only have one input that contains a filter expression".into(),
            )),
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
        // two possible paths
        // all inputs may be final aggregations
        // or they may be expression that can work on groups but not yet produce an aggregation

        // we first collect the inputs
        // if any of the input aggregations yields None, we return None as well
        // we check this by comparing the length of the inputs before and after aggregation
        let mut inputs: Vec<_> = match self.inputs[0].as_agg_expr() {
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
            if inputs.len() == 1 {
                let s = inputs.pop().unwrap();

                match (s.list(), self.collect_groups) {
                    (Ok(ca), true) => {
                        let mut container = vec![Default::default()];
                        let name = s.name();

                        let mut ca: ListChunked = ca
                            .into_iter()
                            .map(|opt_s| {
                                opt_s.and_then(|s| {
                                    container[0] = s;
                                    self.function.call_udf(&mut container).ok()
                                })
                            })
                            .collect();
                        ca.rename(name);
                        Ok(Some(ca.into_series()))
                    }
                    _ => self.function.call_udf(&mut [s]).map(Some),
                }
            } else {
                match (inputs[0].list(), self.collect_groups) {
                    (Ok(_), true) => {
                        // container that will hold the arguments &[Series]
                        let mut args = Vec::with_capacity(inputs.len());
                        let takers: Vec<_> = inputs
                            .iter()
                            .map(|s| s.list().unwrap().take_rand())
                            .collect();
                        let mut ca: ListChunked = (0..inputs[0].len())
                            .map(|i| {
                                args.clear();

                                takers.iter().for_each(|taker| {
                                    if let Some(s) = taker.get(i) {
                                        args.push(s);
                                    }
                                });
                                if args.len() == takers.len() {
                                    self.function.call_udf(&mut args).ok()
                                } else {
                                    None
                                }
                            })
                            .collect();
                        ca.rename(inputs[0].name());
                        Ok(Some(ca.into_series()))
                    }
                    _ => self.function.call_udf(&mut inputs).map(Some),
                }
            }
        } else {
            Ok(None)
        }
    }
}
