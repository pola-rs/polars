use polars_utils::arena::{Arena, Node};

use crate::prelude::{AExpr, ALogicalPlan};

pub trait Visitor {
    type Item;

    fn visit_alp(
        &self,
        node: Node,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Self::Item;
}
