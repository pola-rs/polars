use polars_utils::vec::CapacityByFactor;

use super::*;
use crate::constants::CSE_REPLACED;
use crate::logical_plan::projection_expr::ProjectionExprs;
use crate::prelude::visitor::ALogicalPlanNode;

mod identifier_impl {
    use std::hash::{Hash, Hasher};

    use ahash::RandomState;
    use polars_core::hashing::_boost_hash_combine;

    use super::*;
    /// Identifier that shows the sub-expression path.
    /// Must implement hash and equality and ideally
    /// have little collisions
    /// We will do a full expression comparison to check if the
    /// expressions with equal identifiers are truly equal
    #[derive(Clone)]
    struct Identifier<'a> {
        inner: Option<u64>,
        last_node: Option<ALogicalPlanNode>,
        hb: RandomState,
        expr_arena: &'a Arena<AExpr>
    }

    impl PartialEq<Self> for Identifier<'_> {
        fn eq(&self, other: &Self) -> bool {
            self.inner == other.inner && match (self.last_node, other.last_node) {
                (None, None) => true,
                (Some(l), Some(r)) => {
                    l.hashable_and_cmp(self.expr_arena) == r.hashable_and_cmp(self.expr_arena)
                },
                _ => false
            }
        }
    }

    impl Eq for Identifier<'_> {}

    impl Hash for Identifier<'_> {
        fn hash<H: Hasher>(&self, state: &mut H) {
            state.write_u64(self.inner.unwrap_or(0))
        }
    }

    impl<'a> Identifier<'a> {
        pub fn new(expr_arena: &'a Arena<AExpr>) -> Self {
            Self {
                inner: None,
                last_node: None,
                hb: RandomState::with_seed(0),
                expr_arena
            }
        }

        pub fn alp_node(&self) -> ALogicalPlanNode {
            self.last_node.unwrap()
        }

        pub fn is_valid(&self) -> bool {
            self.inner.is_some()
        }

        pub fn materialize(&self) -> String {
            format!("{}{}", CSE_REPLACED, self.inner.unwrap_or(0))
        }

        pub fn combine(&mut self, other: &Identifier) {
            let inner = match (self.inner, other.inner) {
                (Some(l), Some(r)) => _boost_hash_combine(l, r),
                (None, Some(r)) => r,
                (Some(l), None) => l,
                _ => return,
            };
            self.inner = Some(inner);
        }

        pub fn add_alp_node(&self, alp: &ALogicalPlanNode, expr_arena: &Arena<AExpr>) -> Self {
            let hashed = self.hb.hash_one(alp.hashable_and_cmp(expr_arena));
            let inner = Some(
                self.inner
                    .map_or(hashed, |l| _boost_hash_combine(l, hashed)),
            );
            Self {
                inner,
                last_node: Some(*alp),
                hb: self.hb.clone(),
                expr_arena: self.expr_arena
            }
        }
    }
}
use identifier_impl::*;
