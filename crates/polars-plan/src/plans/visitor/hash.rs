use std::hash::{Hash, Hasher};

use polars_utils::arena::{Arena, Node};

use super::*;
use crate::plans::{AExpr, ExpressionHasher, IR};
use crate::prelude::ExprIR;

impl IRNode {
    pub(crate) fn hashable_and_cmp<'a>(
        &'a self,
        lp_arena: &'a Arena<IR>,
        expr_arena: &'a Arena<AExpr>,
    ) -> IRHashWrap<'a> {
        IRHashWrap {
            node: self.node(),
            lp_arena,
            expr_arena,
        }
    }
}

pub(crate) struct IRHashWrap<'a> {
    node: Node,
    lp_arena: &'a Arena<IR>,
    expr_arena: &'a Arena<AExpr>,
}

impl<'a> IRHashWrap<'a> {
    pub(crate) fn new(node: Node, lp_arena: &'a Arena<IR>, expr_arena: &'a Arena<AExpr>) -> Self {
        Self {
            node,
            lp_arena,
            expr_arena,
        }
    }
}

struct TraverseAndHashExpr<'a> {
    expr_arena: &'a Arena<AExpr>,
}

impl ExpressionHasher for TraverseAndHashExpr<'_> {
    fn hash_expr<H: Hasher>(&self, expr: &ExprIR, state: &mut H) {
        expr.traverse_and_hash(self.expr_arena, state);
    }
}

impl Hash for IRHashWrap<'_> {
    // This hashes the variant, not the whole plan
    fn hash<H: Hasher>(&self, state: &mut H) {
        let alp = self.lp_arena.get(self.node);
        alp.shallow_hash(
            state,
            &TraverseAndHashExpr {
                expr_arena: self.expr_arena,
            },
        );
    }
}
