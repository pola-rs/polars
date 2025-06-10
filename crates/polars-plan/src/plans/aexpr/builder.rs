use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::DataType;
use polars_utils::arena::{Arena, Node};

use super::AExpr;
use crate::dsl::Operator;

#[derive(Clone, Copy)]
pub struct AExprBuilder {
    node: Node,
}

impl AExprBuilder {
    pub fn new_from_node(node: Node) -> Self {
        Self { node }
    }

    pub fn cast(self, dtype: DataType, arena: &mut Arena<AExpr>) -> Self {
        Self {
            node: arena.add(AExpr::Cast {
                expr: self.node,
                dtype,
                options: CastOptions::Strict,
            }),
        }
    }

    pub fn binary_op(
        self,
        other: impl IntoAExprBuilder,
        op: Operator,
        arena: &mut Arena<AExpr>,
    ) -> Self {
        Self {
            node: arena.add(AExpr::BinaryExpr {
                left: self.node,
                op,
                right: other.into_aexpr_builder().node,
            }),
        }
    }

    pub fn logical_and(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::LogicalAnd, arena)
    }

    pub fn logical_or(self, other: impl IntoAExprBuilder, arena: &mut Arena<AExpr>) -> Self {
        self.binary_op(other, Operator::LogicalOr, arena)
    }

    pub fn node(self) -> Node {
        self.node
    }
}

pub trait IntoAExprBuilder {
    fn into_aexpr_builder(self) -> AExprBuilder;
}

impl IntoAExprBuilder for Node {
    fn into_aexpr_builder(self) -> AExprBuilder {
        AExprBuilder { node: self }
    }
}

impl IntoAExprBuilder for AExprBuilder {
    fn into_aexpr_builder(self) -> AExprBuilder {
        self
    }
}
