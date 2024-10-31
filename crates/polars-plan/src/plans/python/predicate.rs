use polars_core::error::polars_err;
use polars_core::prelude::PolarsResult;

use crate::prelude::*;

fn accept_as_io_predicate(e: &Expr) -> bool {
    const LIMIT: usize = 1 << 16;
    match e {
        Expr::Literal(lv) => match lv {
            LiteralValue::Binary(v) => v.len() <= LIMIT,
            LiteralValue::String(v) => v.len() <= LIMIT,
            LiteralValue::Series(s) => s.estimated_size() < LIMIT,
            // Don't accept dynamic types
            LiteralValue::Int(_) => false,
            LiteralValue::Float(_) => false,
            _ => true,
        },
        Expr::Wildcard | Expr::Column(_) => true,
        Expr::BinaryExpr { left, right, .. } => {
            accept_as_io_predicate(left) && accept_as_io_predicate(right)
        },
        Expr::Ternary {
            truthy,
            falsy,
            predicate,
        } => {
            accept_as_io_predicate(truthy)
                && accept_as_io_predicate(falsy)
                && accept_as_io_predicate(predicate)
        },
        Expr::Alias(_, _) => true,
        Expr::Function {
            function, input, ..
        } => {
            match function {
                // we already checked if streaming, so we can all functions
                FunctionExpr::Boolean(_) | FunctionExpr::BinaryExpr(_) | FunctionExpr::Coalesce => {
                },
                #[cfg(feature = "log")]
                FunctionExpr::Entropy { .. }
                | FunctionExpr::Log { .. }
                | FunctionExpr::Log1p { .. }
                | FunctionExpr::Exp { .. } => {},
                #[cfg(feature = "abs")]
                FunctionExpr::Abs => {},
                #[cfg(feature = "trigonometry")]
                FunctionExpr::Atan2 => {},
                #[cfg(feature = "round_series")]
                FunctionExpr::Clip { .. } => {},
                #[cfg(feature = "fused")]
                FunctionExpr::Fused(_) => {},
                _ => return false,
            }
            input.iter().all(accept_as_io_predicate)
        },
        _ => false,
    }
}

pub fn serialize(expr: &Expr) -> PolarsResult<Option<Vec<u8>>> {
    if !accept_as_io_predicate(expr) {
        return Ok(None);
    }
    let mut buf = vec![];
    ciborium::into_writer(expr, &mut buf)
        .map_err(|_| polars_err!(ComputeError: "could not serialize: {}", expr))?;

    Ok(Some(buf))
}
