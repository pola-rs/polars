use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::{prelude::*, POOL};
use std::borrow::Cow;
use std::sync::Arc;

/// In the aggregation of a binary expression, only one expression can modify the size of the groups
/// with a filter operation otherwise the aggregations will produce flawed results.
pub(crate) fn binary_check_group_tuples<'a>(
    out: Series,
    groups_a: Cow<'a, GroupTuples>,
    groups_b: Cow<'a, GroupTuples>,
) -> Result<(Series, Cow<'a, GroupTuples>)> {
    match (groups_a, groups_b) {
        (Cow::Owned(_), Cow::Owned(_)) => Err(PolarsError::InvalidOperation(
            "Cannot apply two filters in a binary expression".into(),
        )),
        (Cow::Borrowed(a), Cow::Borrowed(b)) => {
            if !std::ptr::eq(a, b) {
                Err(PolarsError::ValueError(
                    "filter predicates do not originate from same filter operation".into(),
                ))
            } else {
                Ok((out, Cow::Borrowed(a)))
            }
        }
        (Cow::Owned(a), _) => Ok((out, Cow::Owned(a))),
        (_, Cow::Owned(a)) => Ok((out, Cow::Owned(a))),
    }
}

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
        let (lhs, rhs) = POOL.install(|| {
            rayon::join(
                || self.left.evaluate(df, state),
                || self.right.evaluate(df, state),
            )
        });
        apply_operator(&lhs?, &rhs?, self.op)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<(Series, Cow<'a, GroupTuples>)> {
        let (result_a, result_b) = POOL.install(|| {
            rayon::join(
                || self.left.evaluate_on_groups(df, groups, state),
                || self.right.evaluate_on_groups(df, groups, state),
            )
        });
        let (series_a, groups_a) = result_a?;
        let (series_b, groups_b) = result_b?;

        let out = apply_operator(&series_a, &series_b, self.op)?;
        binary_check_group_tuples(out, groups_a, groups_b)
    }

    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        todo!()
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
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
