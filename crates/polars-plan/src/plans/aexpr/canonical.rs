use std::hash::{DefaultHasher, Hash, Hasher};

use hashbrown::HashTable;
use polars_core::prelude::{InitHashMaps as _, PlIndexMap};
use polars_utils::arena::{Arena, Node};

#[cfg(feature = "cse")]
use crate::plans::ExpressionHasher;
use crate::plans::{AExpr, ExprIR, ExpressionComparator};

/// Identifies an `AExpr` up to structural equality: two nodes get the same
/// `CanonicalExprId` if and only if they represent the same expression.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CanonicalExprId(u32);

impl CanonicalExprId {
    fn as_u32(self) -> u32 {
        self.0
    }
}

struct CanonicalExprEntry {
    representative: Node,
    child_ids: Vec<CanonicalExprId>,
    id: CanonicalExprId,
}

/// Assigns [`CanonicalExprId`]s to `AExpr` nodes.
pub struct CanonicalExprMap {
    deduplication_map: HashTable<CanonicalExprEntry>,
    cache: PlIndexMap<Node, CanonicalExprId>,
    representatives: Vec<Node>,
}

impl CanonicalExprMap {
    pub fn new() -> Self {
        Self {
            deduplication_map: HashTable::new(),
            cache: PlIndexMap::new(),
            representatives: Vec::new(),
        }
    }

    /// Forgets all resolved expressions, retaining the allocated capacity.
    ///
    /// This must be called whenever nodes are rewritten in place in the expression arena the map was
    /// populated from, as the cached ids no longer describe those nodes. Adding nodes to the arena is
    /// fine. All previously returned [`CanonicalExprId`]s become meaningless.
    pub fn clear(&mut self) {
        self.deduplication_map.clear();
        self.cache.clear();
        self.representatives.clear();
    }

    /// Returns the representative node for `id`: the first node of its structural equivalence
    /// class that was passed to [`Self::resolve`].
    pub fn representative(&self, id: CanonicalExprId) -> Node {
        let index =
            id.0.checked_sub(1)
                .expect("canonical expression IDs start at one") as usize;
        self.representatives[index]
    }

    /// Returns the id of `node`, resolving its children recursively.
    pub fn resolve(&mut self, node: Node, expr_arena: &Arena<AExpr>) -> CanonicalExprId {
        if let Some(id) = self.cache.get(&node) {
            return *id;
        }

        // This cannot use the regular AExpr tree walker: canonical equality includes all
        // children, including expressions evaluated in a nested context by `AExpr::Eval`.
        let mut children = Vec::new();
        let mut stack = vec![(node, false)];

        while let Some((node, post_visit)) = stack.pop() {
            if self.cache.contains_key(&node) {
                continue;
            }

            if post_visit {
                children.clear();
                expr_arena.get(node).children_rev(&mut children);
                let child_ids = children.iter().map(|child| self.get(*child)).collect();
                let id = self.resolve_single(node, child_ids, expr_arena);
                self.cache.insert(node, id);
            } else {
                stack.push((node, true));
                children.clear();
                expr_arena.get(node).children_rev(&mut children);
                stack.extend(children.iter().map(|&child| (child, false)));
            }
        }

        self.get(node)
    }

    fn resolve_single(
        &mut self,
        node: Node,
        child_ids: Vec<CanonicalExprId>,
        expr_arena: &Arena<AExpr>,
    ) -> CanonicalExprId {
        let hash = combined_hash(node, &child_ids, expr_arena);
        let next_id = CanonicalExprId(1 + self.deduplication_map.len() as u32);
        let id = self
            .deduplication_map
            .entry(
                hash,
                |other| {
                    child_ids == other.child_ids
                        && expr_arena
                            .get(node)
                            .is_expr_equal_shallow(expr_arena.get(other.representative))
                },
                |other| combined_hash(other.representative, &other.child_ids, expr_arena),
            )
            .or_insert(CanonicalExprEntry {
                representative: node,
                child_ids,
                id: next_id,
            })
            .get()
            .id;

        if id == next_id {
            self.representatives.push(node);
        }
        id
    }

    fn get(&self, node: Node) -> CanonicalExprId {
        *self.cache.get(&node).unwrap_or_else(|| {
            panic!("expression node {node:?} was not resolved by CanonicalExprMap")
        })
    }
}

impl Default for CanonicalExprMap {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpressionComparator for CanonicalExprMap {
    fn equals(&self, lhs: &ExprIR, rhs: &ExprIR) -> bool {
        self.get(lhs.node()) == self.get(rhs.node())
            && lhs.output_name_inner() == rhs.output_name_inner()
    }
}

#[cfg(feature = "cse")]
impl ExpressionHasher for CanonicalExprMap {
    fn hash_expr<H: Hasher>(&self, expr: &ExprIR, state: &mut H) {
        state.write_u32(self.get(expr.node()).as_u32());
        expr.output_name_inner().hash(state);
    }
}

/// Hashes the node itself, without descending into its children, combined with
/// the ids of those children.
fn combined_hash(node: Node, child_ids: &[CanonicalExprId], expr_arena: &Arena<AExpr>) -> u64 {
    let mut hasher = DefaultHasher::new();
    expr_arena.get(node).hash(&mut hasher);
    for child_id in child_ids {
        hasher.write_u32(child_id.as_u32());
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use polars_core::scalar::Scalar;

    use super::*;
    use crate::dsl::{EvalVariant, Operator};
    use crate::plans::LiteralValue;
    #[cfg(feature = "dtype-struct")]
    use crate::plans::{ExprIR, OutputName};

    fn add_literal(arena: &mut Arena<AExpr>, left: Node, value: i64) -> Node {
        let right = arena.add(AExpr::Literal(LiteralValue::Scalar(Scalar::from(value))));
        arena.add(AExpr::BinaryExpr {
            left,
            op: Operator::Plus,
            right,
        })
    }

    #[test]
    fn resolves_structural_equivalence_and_representatives() {
        let mut arena = Arena::new();

        let left_a = arena.add(AExpr::Column("a".into()));
        let left_b = arena.add(AExpr::Column("b".into()));
        let left = arena.add(AExpr::BinaryExpr {
            left: left_a,
            op: Operator::Plus,
            right: left_b,
        });

        let right_a = arena.add(AExpr::Column("a".into()));
        let right_b = arena.add(AExpr::Column("b".into()));
        let right = arena.add(AExpr::BinaryExpr {
            left: right_a,
            op: Operator::Plus,
            right: right_b,
        });

        let different = arena.add(AExpr::BinaryExpr {
            left: right_a,
            op: Operator::Minus,
            right: right_b,
        });

        let mut map = CanonicalExprMap::new();
        let left_id = map.resolve(left, &arena);
        assert_eq!(left_id, map.resolve(right, &arena));
        assert_eq!(map.representative(left_id), left);
        assert_ne!(left_id, map.resolve(different, &arena));
    }

    #[test]
    fn includes_nested_evaluation_children() {
        let mut arena = Arena::new();
        let input = arena.add(AExpr::Column("values".into()));
        let element = arena.add(AExpr::Element);
        let add_one = add_literal(&mut arena, element, 1);
        let add_two = add_literal(&mut arena, element, 2);

        let first = arena.add(AExpr::Eval {
            expr: input,
            evaluation: add_one,
            variant: EvalVariant::List,
        });
        let second = arena.add(AExpr::Eval {
            expr: input,
            evaluation: add_two,
            variant: EvalVariant::List,
        });

        let mut map = CanonicalExprMap::new();
        assert_ne!(map.resolve(first, &arena), map.resolve(second, &arena));

        #[cfg(feature = "dtype-struct")]
        {
            let struct_input = arena.add(AExpr::Column("struct".into()));
            let field = arena.add(AExpr::StructField("field".into()));
            let add_one = add_literal(&mut arena, field, 1);
            let add_two = add_literal(&mut arena, field, 2);
            let output_name = OutputName::Alias("field".into());

            let first = arena.add(AExpr::StructEval {
                expr: struct_input,
                evaluation: vec![ExprIR::new(add_one, output_name.clone())],
            });
            let second = arena.add(AExpr::StructEval {
                expr: struct_input,
                evaluation: vec![ExprIR::new(add_two, output_name)],
            });

            assert_ne!(map.resolve(first, &arena), map.resolve(second, &arena));
        }
    }
}
