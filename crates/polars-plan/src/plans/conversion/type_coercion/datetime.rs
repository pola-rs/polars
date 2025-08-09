use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_utils::arena::Arena;

use crate::plans::{AExpr, ExprIR, IRFunctionExpr, IRTemporalFunction, LiteralValue};
use crate::prelude::FunctionOptions;

/// Cast a date or datetime node to a supertype.
///
/// If the target datetime type has a timezone, then:
///   * If the source is a Date, we first cast to naive datetime, then replace the time zone.
///   * If the source is a Datetime and is naive, we replace the time zone.
///   * If the source is a Datetime with a different time zone, we convert time zone.
///
pub fn coerce_dt(
    from_type: &DataType,
    to_type: &DataType,
    expr: &mut ExprIR,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<()> {
    if from_type != to_type {
        if let &DataType::Datetime(tu, Some(ref from_tz)) = to_type {
            match from_type {
                DataType::Date => {
                    expr.set_node(expr_arena.add(AExpr::Cast {
                        expr: expr.node(),
                        dtype: DataType::Datetime(tu, None),
                        options: CastOptions::Strict,
                    }));
                    replace_tz(expr, to_type, expr_arena)?;
                    return Ok(());
                },
                DataType::Datetime(_, None) => {
                    replace_tz(expr, to_type, expr_arena)?;
                },
                DataType::Datetime(_, Some(to_tz)) if from_tz != to_tz => {
                    convert_tz(expr, to_type, expr_arena)?;
                },
                _ => (),
            }
        }
        expr.set_node(expr_arena.add(AExpr::Cast {
            expr: expr.node(),
            dtype: to_type.clone(),
            options: CastOptions::Strict,
        }));
        expr.set_dtype(to_type.clone());
    }
    Ok(())
}

pub(super) fn replace_tz(
    e: &mut ExprIR,
    dtype: &DataType,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<()> {
    let replacement_expr = match &dtype {
        // Wrap our node in a ReplaceTimezone node
        &DataType::Datetime(_, tz @ Some(_)) => AExpr::Function {
            input: {
                // We must add the ambiguous input argument.
                let scalar = Scalar::new(DataType::String, AnyValue::from("raise"));
                let node = expr_arena.add(AExpr::Literal(LiteralValue::Scalar(scalar)));
                let ambiguous = ExprIR::from_node(node, expr_arena);
                vec![e.clone(), ambiguous]
            },
            function: IRFunctionExpr::TemporalExpr(IRTemporalFunction::ReplaceTimeZone(
                tz.clone(),
                NonExistent::Raise,
            )),
            options: FunctionOptions::elementwise(),
        },
        dt => polars_bail!(ComputeError: "cannot replace time zone of dtype {:?}", dt),
    };

    e.set_node(expr_arena.add(replacement_expr));
    e.set_dtype(dtype.clone());
    Ok(())
}

pub(super) fn convert_tz(
    e: &mut ExprIR,
    dtype: &DataType,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<()> {
    let replacement_expr = match &dtype {
        // Wrap our node in a ReplaceTimezone node
        &DataType::Datetime(_, Some(tz)) => AExpr::Function {
            input: vec![e.clone()],
            function: IRFunctionExpr::TemporalExpr(IRTemporalFunction::ConvertTimeZone(tz.clone())),
            options: FunctionOptions::elementwise(),
        },
        dt => polars_bail!(ComputeError: "cannot convert time zone of dtype {:?}", dt),
    };

    e.set_node(expr_arena.add(replacement_expr));
    e.set_dtype(dtype.clone());
    Ok(())
}
