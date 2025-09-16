use super::*;

pub fn is_scalar_ae(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    expr_arena.get(node).is_scalar(expr_arena)
}
