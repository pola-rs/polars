//! Lowering of [`SqlFunction`]s, once the operand dtypes are known.

use polars_core::utils::try_get_supertype;

use super::*;
use crate::plans::AExprBuilder;

pub(super) fn lower_sql_function(
    function: SqlFunction,
    mut e: Vec<ExprIR>,
    ctx: &mut ExprToIRContext,
) -> PolarsResult<Node> {
    match function {
        SqlFunction::Binary(op) => lower_binary(op, e, ctx),
        #[cfg(feature = "is_in")]
        SqlFunction::IsIn { nulls_equal } => {
            let needle = e[0].dtype(ctx.schema, ctx.arena)?.clone();
            adapt_decimal_literal(&mut e[1], &needle, ctx)?;
            let function = IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { nulls_equal });
            Ok(AExprBuilder::function(e, function, ctx.arena).node())
        },
        SqlFunction::ToFloat => {
            let mut input = e.pop().unwrap();
            if input.dtype(ctx.schema, ctx.arena)?.is_decimal() {
                cast_to(&mut input, &DataType::Float64, ctx)?;
            }
            Ok(input.node())
        },
        #[cfg(feature = "round_series")]
        SqlFunction::Round { decimals } => {
            let mode = if e[0].dtype(ctx.schema, ctx.arena)?.is_decimal() {
                RoundMode::HalfAwayFromZero
            } else {
                RoundMode::default()
            };
            let function = IRFunctionExpr::Round { decimals, mode };
            Ok(AExprBuilder::function(e, function, ctx.arena).node())
        },
    }
}

fn lower_binary(
    op: SqlBinaryOp,
    mut e: Vec<ExprIR>,
    ctx: &mut ExprToIRContext,
) -> PolarsResult<Node> {
    let left = e[0].dtype(ctx.schema, ctx.arena)?.clone();
    let right = e[1].dtype(ctx.schema, ctx.arena)?.clone();
    adapt_decimal_literal(&mut e[0], &right, ctx)?;
    adapt_decimal_literal(&mut e[1], &left, ctx)?;

    let operator = match op {
        SqlBinaryOp::Plus => Operator::Plus,
        SqlBinaryOp::Minus => Operator::Minus,
        SqlBinaryOp::Eq => Operator::Eq,
        SqlBinaryOp::EqMissing => Operator::EqValidity,
        SqlBinaryOp::Lt => Operator::Lt,
        SqlBinaryOp::LtEq => Operator::LtEq,
        SqlBinaryOp::Gt => Operator::Gt,
        SqlBinaryOp::GtEq => Operator::GtEq,
        SqlBinaryOp::Mul | SqlBinaryOp::Div => {
            if let Some(function) = decimal_arith(op, &e, ctx)? {
                return Ok(AExprBuilder::function(e, function, ctx.arena).node());
            }
            if op == SqlBinaryOp::Mul {
                Operator::Multiply
            } else {
                Operator::TrueDivide
            }
        },
        SqlBinaryOp::Rem | SqlBinaryOp::IntDiv => {
            if let Some(node) = trunc_arith(op, &mut e, ctx)? {
                return Ok(node);
            }
            if op == SqlBinaryOp::Rem {
                Operator::Modulus
            } else {
                Operator::FloorDivide
            }
        },
    };
    Ok(ctx.arena.add(AExpr::BinaryExpr {
        left: e[0].node(),
        op: operator,
        right: e[1].node(),
    }))
}

/// Whether `node` only takes literal values: a literal, or a `CASE` of them.
fn is_literal_valued(node: Node, arena: &Arena<AExpr>) -> bool {
    match arena.get(node) {
        AExpr::Literal(_) => true,
        AExpr::Ternary { truthy, falsy, .. } => {
            is_literal_valued(*truthy, arena) && is_literal_valued(*falsy, arena)
        },
        _ => false,
    }
}

/// A literal-valued decimal operand (see [`is_literal_valued`]) combined with a float takes
/// the float's type, as a float literal would. Decimal columns keep the usual supertype.
fn adapt_decimal_literal(
    e: &mut ExprIR,
    other: &DataType,
    ctx: &mut ExprToIRContext,
) -> PolarsResult<()> {
    if !other.is_float() {
        return Ok(());
    }
    let target = match e.dtype(ctx.schema, ctx.arena)? {
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, _) => other.clone(),
        DataType::List(inner) if inner.is_decimal() => DataType::List(Box::new(other.clone())),
        _ => return Ok(()),
    };
    if is_literal_valued(e.node(), ctx.arena) {
        cast_to(e, &target, ctx)?;
    }
    Ok(())
}

fn cast_to(e: &mut ExprIR, dtype: &DataType, ctx: &mut ExprToIRContext) -> PolarsResult<()> {
    if e.dtype(ctx.schema, ctx.arena)? != dtype {
        let node = ctx.arena.add(AExpr::Cast {
            expr: e.node(),
            dtype: dtype.clone(),
            options: CastOptions::Strict,
        });
        *e = ExprIR::new(node, e.output_name_inner().clone());
    }
    Ok(())
}

/// SQL `%` and `DIV` of numerics truncate toward zero. The operands are cast to their
/// supertype, or with a decimal, integers to `Decimal(38, 0)`. `DIV` gives the quotient as
/// `Decimal(38, 0)` for decimals and `Int64` otherwise, raising if it doesn't fit. `None`
/// for non-numeric operands.
fn trunc_arith(
    op: SqlBinaryOp,
    e: &mut [ExprIR],
    ctx: &mut ExprToIRContext,
) -> PolarsResult<Option<Node>> {
    let numeric = |dt: &DataType| {
        dt.is_primitive_numeric()
            || dt.is_decimal()
            || matches!(
                dt,
                DataType::Unknown(UnknownKind::Int(_) | UnknownKind::Float)
            )
    };
    let left = e[0].dtype(ctx.schema, ctx.arena)?.clone();
    let right = e[1].dtype(ctx.schema, ctx.arena)?.clone();
    if !(numeric(&left) && numeric(&right)) {
        return Ok(None);
    }
    let is_float =
        |dt: &DataType| dt.is_float() || matches!(dt, DataType::Unknown(UnknownKind::Float));
    // A float with a decimal is approximate, as for the other operators.
    let decimal =
        (left.is_decimal() || right.is_decimal()) && !(is_float(&left) || is_float(&right));
    #[cfg(feature = "dtype-decimal")]
    if decimal {
        let int_decimal = DataType::Decimal(polars_compute::decimal::DEC128_MAX_PREC, 0);
        for (i, dt) in [&left, &right].into_iter().enumerate() {
            if !dt.is_decimal() {
                cast_to(&mut e[i], &int_decimal, ctx)?;
            }
        }
    }
    let trunc_op = if op == SqlBinaryOp::Rem {
        TruncArithOp::Rem
    } else {
        TruncArithOp::IntDiv
    };
    if !decimal {
        let mut supertype = try_get_supertype(&left, &right)?.materialize_unknown(true)?;
        // The Int64 quotient of narrower integers can exceed their type: -128 DIV -1.
        if trunc_op == TruncArithOp::IntDiv
            && supertype.is_integer()
            && !matches!(
                supertype,
                DataType::Int64 | DataType::UInt64 | DataType::Int128 | DataType::UInt128
            )
        {
            supertype = DataType::Int64;
        }
        for x in e.iter_mut() {
            cast_to(x, &supertype, ctx)?;
        }
    }
    let node =
        AExprBuilder::function(e.to_vec(), IRFunctionExpr::TruncArith(trunc_op), ctx.arena).node();
    // An exact quotient is Decimal(38, 0), as in PostgreSQL; others are Int64.
    if trunc_op == TruncArithOp::Rem || decimal {
        return Ok(Some(node));
    }
    Ok(Some(ctx.arena.add(AExpr::Cast {
        expr: node,
        dtype: DataType::Int64,
        options: CastOptions::Strict,
    })))
}

/// SQL `*` and `/` of exact numerics (decimal or integer, at least one decimal), instead of
/// the `max(s1, s2)` scale of the expression API. `*` keeps scale `s1 + s2` as the SQL standard
/// requires. `/` uses Snowflake's scale `max(s1, min(s1 + 6, 12))`, which depends only on the
/// scales, since Polars widens most decimal results to precision 38. `None` for ordinary
/// arithmetic.
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
            SqlBinaryOp::Div => (DecimalArithOp::Div, s1.max((s1 + 6).min(12))),
            _ => unreachable!(),
        };
        Ok(Some(IRFunctionExpr::DecimalArith { op, scale }))
    }
    #[cfg(not(feature = "dtype-decimal"))]
    {
        let _ = (op, e, ctx);
        Ok(None)
    }
}
