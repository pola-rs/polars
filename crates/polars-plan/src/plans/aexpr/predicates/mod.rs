mod column_expr;
mod skip_batches;

use std::borrow::Cow;

pub use column_expr::*;
use polars_core::prelude::DataType;
use polars_core::schema::Schema;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
pub use skip_batches::*;

use super::evaluate::{constant_evaluate, into_column};
use super::{AExpr, LiteralValue};

#[allow(clippy::type_complexity)]
fn get_binary_expr_col_and_lv<'a>(
    left: Node,
    right: Node,
    expr_arena: &'a Arena<AExpr>,
    schema: &Schema,
) -> Option<(
    (&'a PlSmallStr, Node),
    (Option<Cow<'a, LiteralValue>>, Node),
)> {
    match (
        into_column(left, expr_arena, schema, 0),
        into_column(right, expr_arena, schema, 0),
        constant_evaluate(left, expr_arena, schema, 0),
        constant_evaluate(right, expr_arena, schema, 0),
    ) {
        (Some(col), _, _, Some(lv)) => Some(((col, left), (lv, right))),
        (_, Some(col), Some(lv), _) => Some(((col, right), (lv, left))),
        _ => None,
    }
}

#[allow(clippy::type_complexity)]
fn get_binary_expr_col_and_lv_casted<'a>(
    left: Node,
    right: Node,
    expr_arena: &'a Arena<AExpr>,
    schema: &Schema,
) -> Option<(
    (&'a PlSmallStr, Node, Option<&'a DataType>),
    (Option<Cow<'a, LiteralValue>>, Node),
)> {
    if let Some(lv) = constant_evaluate(left, expr_arena, schema, 0) {
        let (col, col_node, dtype) = if let AExpr::Cast {
            expr: inner,
            dtype,
            options: _,
        } = expr_arena.get(right)
        {
            (
                into_column(*inner, expr_arena, schema, 0)?,
                *inner,
                Some(dtype),
            )
        } else {
            (into_column(right, expr_arena, schema, 0)?, right, None)
        };

        Some(((col, col_node, dtype), (lv, left)))
    } else if let Some(lv) = constant_evaluate(right, expr_arena, schema, 0) {
        let (col, col_node, dtype) = if let AExpr::Cast {
            expr: inner,
            dtype,
            options: _,
        } = expr_arena.get(left)
        {
            (
                into_column(*inner, expr_arena, schema, 0)?,
                *inner,
                Some(dtype),
            )
        } else {
            (into_column(left, expr_arena, schema, 0)?, left, None)
        };

        Some(((col, col_node, dtype), (lv, right)))
    } else {
        None
    }
}
