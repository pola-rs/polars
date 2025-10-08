use super::*;

pub fn is_scalar_ae(node: Node, arena: &Arena<AExpr>) -> bool {
    arena.get(node).is_scalar(arena)
}

pub fn is_length_preserving_ae(node: Node, arena: &Arena<AExpr>) -> bool {
    arena.get(node).is_length_preserving(arena)
}
