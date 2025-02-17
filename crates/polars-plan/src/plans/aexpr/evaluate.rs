use std::borrow::Cow;

use polars_core::schema::Schema;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use super::{aexpr_to_leaf_names_iter, AExpr, LiteralValue};

pub fn constant_evaluate<'a>(
    e: Node,
    expr_arena: &'a Arena<AExpr>,
    _schema: &Schema,
    _depth: usize,
) -> Option<Option<Cow<'a, LiteralValue>>> {
    match expr_arena.get(e) {
        AExpr::Literal(lv) => Some(Some(Cow::Borrowed(lv))),
        _ => {
            if aexpr_to_leaf_names_iter(e, expr_arena).next().is_none() {
                Some(None)
            } else {
                None
            }
        },
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
