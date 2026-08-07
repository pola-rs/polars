use std::hash::{DefaultHasher, Hash, Hasher};

use hashbrown::HashTable;
use polars_core::prelude::{InitHashMaps as _, PlIndexMap};
use polars_utils::arena::{Arena, Node};

use crate::plans::{AExpr, ExprIR, ExpressionComparator, ExpressionHasher};

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
}

impl CanonicalExprMap {
    pub fn new() -> Self {
        Self {
            deduplication_map: HashTable::new(),
            cache: PlIndexMap::new(),
        }
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
        self.deduplication_map
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
            .id
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
