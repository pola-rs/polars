//! Lowering of [`SqlFunction`]s, once the operand dtypes are known.

use super::*;
use crate::plans::AExprBuilder;

pub(super) fn lower_sql_function(
    function: SqlFunction,
    e: Vec<ExprIR>,
    ctx: &mut ExprToIRContext,
) -> PolarsResult<Node> {
    match function {
        SqlFunction::Binary(op) => lower_binary(op, e, ctx),
    }
}

fn lower_binary(op: SqlBinaryOp, e: Vec<ExprIR>, ctx: &mut ExprToIRContext) -> PolarsResult<Node> {
    if let Some(function) = decimal_arith(op, &e, ctx)? {
        return Ok(AExprBuilder::function(e, function, ctx.arena).node());
    }
    let operator = match op {
        SqlBinaryOp::Mul => Operator::Multiply,
        SqlBinaryOp::Div => Operator::TrueDivide,
    };
    Ok(ctx.arena.add(AExpr::BinaryExpr {
        left: e[0].node(),
        op: operator,
        right: e[1].node(),
    }))
}

/// SQL `*` and `/` of exact numerics (decimal or integer, at least one decimal): `*` keeps
/// scale `s1 + s2` as the SQL standard requires, and `/` uses scale `max(s1, s2, 6)`,
/// instead of the `max(s1, s2)` of the expression API. `None` for ordinary arithmetic.
fn decimal_arith(
    op: SqlBinaryOp,
    e: &[ExprIR],
    ctx: &mut ExprToIRContext,
) -> PolarsResult<Option<IRFunctionExpr>> {
    #[cfg(feature = "dtype-decimal")]
    {
        let exact_scale = |dt: &DataType| match dt {
            DataType::Decimal(_, s) => Some((*s, true)),
            dt if dt.is_integer() || matches!(dt, DataType::Unknown(UnknownKind::Int(_))) => {
                Some((0, false))
            },
            _ => None,
        };
        let left = exact_scale(e[0].dtype(ctx.schema, ctx.arena)?);
        let right = exact_scale(e[1].dtype(ctx.schema, ctx.arena)?);
        let (Some((s1, dec1)), Some((s2, dec2))) = (left, right) else {
            return Ok(None);
        };
        if !(dec1 || dec2) {
            return Ok(None);
        }
        let max_prec = polars_compute::decimal::DEC128_MAX_PREC;
        let (op, scale) = match op {
            SqlBinaryOp::Mul => {
                let scale = s1 + s2;
                polars_ensure!(
                    scale <= max_prec,
                    InvalidOperation: "numeric value out of range: multiplication result scale {} exceeds {}",
                    scale, max_prec
                );
                (DecimalArithOp::Mul, scale)
            },
            SqlBinaryOp::Div => (DecimalArithOp::Div, s1.max(s2).max(6)),
        };
        Ok(Some(IRFunctionExpr::DecimalArith { op, scale }))
    }
    #[cfg(not(feature = "dtype-decimal"))]
    {
        let _ = (op, e, ctx);
        Ok(None)
    }
}
