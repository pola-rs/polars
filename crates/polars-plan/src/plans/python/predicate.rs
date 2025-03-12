use polars_core::prelude::PolarsResult;
use polars_utils::pl_serialize;
use recursive::recursive;

use crate::prelude::*;

#[recursive]
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
        Expr::AnonymousFunction { input, options, .. } | Expr::Function { input, options, .. } => {
            options.is_elementwise() && input.iter().all(accept_as_io_predicate)
        },
        _ => false,
    }
}

pub fn serialize(expr: &Expr) -> PolarsResult<Option<Vec<u8>>> {
    if !accept_as_io_predicate(expr) {
        return Ok(None);
    }
    let mut buf = vec![];
    pl_serialize::serialize_into_writer::<_, _, true>(&mut buf, expr)?;

    Ok(Some(buf))
}
