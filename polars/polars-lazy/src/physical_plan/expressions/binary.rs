use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::{prelude::*, POOL};
use std::sync::Arc;

pub struct BinaryExpr {
    pub(crate) left: Arc<dyn PhysicalExpr>,
    pub(crate) op: Operator,
    pub(crate) right: Arc<dyn PhysicalExpr>,
    expr: Expr,
}

impl BinaryExpr {
    pub fn new(
        left: Arc<dyn PhysicalExpr>,
        op: Operator,
        right: Arc<dyn PhysicalExpr>,
        expr: Expr,
    ) -> Self {
        Self {
            left,
            op,
            right,
            expr,
        }
    }
}

pub(crate) fn apply_operator(left: &Series, right: &Series, op: Operator) -> Result<Series> {
    match op {
        Operator::Gt => Ok(ChunkCompare::<&Series>::gt(left, right).into_series()),
        Operator::GtEq => Ok(ChunkCompare::<&Series>::gt_eq(left, right).into_series()),
        Operator::Lt => Ok(ChunkCompare::<&Series>::lt(left, right).into_series()),
        Operator::LtEq => Ok(ChunkCompare::<&Series>::lt_eq(left, right).into_series()),
        Operator::Eq => Ok(ChunkCompare::<&Series>::eq(left, right).into_series()),
        Operator::NotEq => Ok(ChunkCompare::<&Series>::neq(left, right).into_series()),
        Operator::Plus => Ok(left + right),
        Operator::Minus => Ok(left - right),
        Operator::Multiply => Ok(left * right),
        Operator::Divide => Ok(left / right),
        Operator::And => Ok((left.bool()? & right.bool()?).into_series()),
        Operator::Or => Ok((left.bool()? | right.bool()?).into_series()),
        Operator::Modulus => Ok(left % right),
    }
}

impl PhysicalExpr for BinaryExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let lhs = self.left.evaluate(df, state)?;
        let rhs = self.right.evaluate(df, state)?;
        apply_operator(&lhs, &rhs, self.op)
    }
    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        todo!()
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
        let a = self.input_a.evaluate(df, state)?;
        let b = self.input_b.evaluate(df, state)?;

        let agg_a = a.agg_list(groups).expect("no data?");
        let agg_b = b.agg_list(groups).expect("no data?");

        // keep track of the output lengths. If they are all unit length,
        // we can explode the array as it would have the same length as the no. of groups
        // if it is not all unit length it should remain a listarray

        let mut all_unit_length = true;

        let ca = agg_a
            .list()
            .unwrap()
            .into_iter()
            .zip(agg_b.list().unwrap())
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

impl PhysicalAggregation for BinaryExpr {
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        match (self.left.as_agg_expr(), self.right.as_agg_expr()) {
            (Ok(left), Err(_)) => {
                let (opt_agg, rhs) = POOL.install(|| {
                    rayon::join(
                        || left.aggregate(df, groups, state),
                        || self.right.evaluate(df, state),
                    )
                });
                opt_agg?
                    .map(|agg| apply_operator(&agg, &rhs?, self.op))
                    .transpose()
            }
            (Err(_), Ok(right)) => {
                let (opt_agg, lhs) = POOL.install(|| {
                    rayon::join(
                        || right.aggregate(df, groups, state),
                        || self.left.evaluate(df, state),
                    )
                });

                opt_agg?
                    .map(|agg| apply_operator(&lhs?, &agg, self.op))
                    .transpose()
            }
            (Ok(left), Ok(right)) => {
                let (left_agg, right_agg) = POOL.install(|| {
                    rayon::join(
                        || left.aggregate(df, groups, state),
                        || right.aggregate(df, groups, state),
                    )
                });
                let right_agg = right_agg?;
                left_agg?
                    .and_then(|left| right_agg.map(|right| apply_operator(&left, &right, self.op)))
                    .transpose()
            }
            (_, _) => Err(PolarsError::Other(
                "both expressions could not be used in an aggregation context.".into(),
            )),
        }
    }
}
