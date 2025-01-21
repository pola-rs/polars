use std::borrow::Cow;

use polars_core::schema::Schema;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use super::{AExpr, LiteralValue};

pub fn constant_evaluate<'a>(
    e: Node,
    expr_arena: &'a Arena<AExpr>,
    _schema: &Schema,
    _depth: usize,
) -> Option<Cow<'a, LiteralValue>> {
    match expr_arena.get(e) {
        AExpr::Literal(lv) => Some(Cow::Borrowed(lv)),
        _ => None,
    }
}

pub fn into_column<'a>(
    e: Node,
    expr_arena: &'a Arena<AExpr>,
    _schema: &Schema,
    _depth: usize,
) -> Option<&'a PlSmallStr> {
    match expr_arena.get(e) {
        AExpr::Column(c) => Some(c),
        _ => None,
    }
}
