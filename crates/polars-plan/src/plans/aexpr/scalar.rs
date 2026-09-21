use super::*;
use crate::plans::projection_height::{ExprProjectionHeight, aexpr_projection_height_rec};

pub fn is_scalar_ae(node: Node, arena: &Arena<AExpr>) -> bool {
    use ExprProjectionHeight as H;

    match aexpr_projection_height_rec(
        node,
        arena,
        &mut Default::default(),
        &mut Default::default(),
    ) {
        H::Scalar => true,
        H::Column | H::Unknown | H::Range => false,
    }
}

/// A literal with exactly one value, including a single row `Series`.
pub fn is_single_literal_ae(node: Node, arena: &Arena<AExpr>) -> bool {
    matches!(arena.get(node), AExpr::Literal(lv) if lv.is_single_value())
}

pub fn is_length_preserving_ae(node: Node, arena: &Arena<AExpr>) -> bool {
    use ExprProjectionHeight as H;

    match aexpr_projection_height_rec(
        node,
        arena,
        &mut Default::default(),
        &mut Default::default(),
    ) {
        H::Column => true,
        H::Scalar | H::Unknown | H::Range => false,
    }
}

pub fn is_known_length_ae(node: Node, arena: &Arena<AExpr>) -> bool {
    use ExprProjectionHeight as H;

    match aexpr_projection_height_rec(
        node,
        arena,
        &mut Default::default(),
        &mut Default::default(),
    ) {
        H::Column | H::Scalar | H::Range => true,
        H::Unknown => false,
    }
}
