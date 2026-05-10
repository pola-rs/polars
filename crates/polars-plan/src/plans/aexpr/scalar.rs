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
        H::Column | H::Unknown => false,
    }
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
        H::Scalar | H::Unknown => false,
    }
}
