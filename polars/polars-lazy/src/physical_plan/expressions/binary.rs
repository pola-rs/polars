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
        #[cfg(feature = "true_div")]
        Operator::TrueDivide => {
            use DataType::*;
            match left.dtype() {
                Date | Datetime | Float32 | Float64 => Ok(left / right),
                _ => Ok(&left.cast(&Float64)? / &right.cast(&Float64)?),
            }
        }
        Operator::And => left.bitand(right),
        Operator::Or => left.bitor(right),
        Operator::Xor => left.bitxor(right),
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
    ) -> Result<AggregationContext<'a>> {
        let (result_a, result_b) = POOL.install(|| {
            rayon::join(
                || self.left.evaluate_on_groups(df, groups, state),
                || self.right.evaluate_on_groups(df, groups, state),
            )
        });
        let mut ac_l = result_a?;
        let ac_r = result_b?;

        if !ac_l.can_combine(&ac_r) {
            return Err(PolarsError::InvalidOperation(
                "\
            cannot combine this binary expression, the groups do not match"
                    .into(),
            ));
        }
        let out = apply_operator(ac_l.flat().as_ref(), ac_r.flat().as_ref(), self.op)?;
        ac_l.combine_groups(ac_r).with_series(out);
        Ok(ac_l)
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
            (_, _) => Err(PolarsError::ComputeError(
                format!(
                    "this binary expression is not an aggregation: {:?}
                pherhaps you should add an aggregation like, '.sum()', '.min()', '.mean()', etc.
                if you really want to collect this binary expression, use `.list()`",
                    self.expr
                )
                .into(),
            )),
        }
    }
}
