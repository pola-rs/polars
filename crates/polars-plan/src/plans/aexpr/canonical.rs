// We don't care about the iteration order in the various caching structures, so we can use
// PlHashMap/PlHashSet
#![allow(clippy::disallowed_types)]

use std::hash::{DefaultHasher, Hash, Hasher};

use hashbrown::HashTable;
use hashbrown::hash_table::Entry;
use polars_utils::aliases::{InitHashMaps as _, PlHashMap, PlHashSet};
use polars_utils::arena::{Arena, Node};
use slotmap::SlotMap;

#[cfg(feature = "cse")]
use crate::plans::ExpressionHasher;
use crate::plans::{AExpr, ExprIR, ExpressionComparator};

slotmap::new_key_type! {
    /// Identifies an `AExpr` up to structural equality: two nodes get the same
    /// `CanonicalExprId` if and only if they represent the same expression.
    pub struct CanonicalExprId;
}

/// Equivalence class of structurally equal expression nodes.
struct CanonicalExprClass {
    members: PlHashSet<Node>,
    child_ids: Vec<CanonicalExprId>,
}

impl CanonicalExprClass {
    /// An arbitrary member.
    fn representative(&self) -> Node {
        *self
            .members
            .iter()
            .next()
            .expect("the equivalence class should be nonempty or dropped altogether")
    }
}

/// Assigns [`CanonicalExprId`]s to `AExpr` nodes.
pub struct CanonicalExprMap {
    deduplication_map: HashTable<CanonicalExprId>,
    cache: PlHashMap<Node, CanonicalExprId>,
    eq_classes: SlotMap<CanonicalExprId, CanonicalExprClass>,
}

impl CanonicalExprMap {
    pub fn new() -> Self {
        Self {
            deduplication_map: HashTable::new(),
            cache: PlHashMap::new(),
            eq_classes: SlotMap::with_key(),
        }
    }

    /// Forgets `node`, dropping its equivalence class if `node` was its last member.
    /// The caller must ensure that no ancestors of `node` are present in the map.
    pub fn remove(&mut self, node: Node, expr_arena: &Arena<AExpr>) {
        let Some(id) = self.cache.remove(&node) else {
            return;
        };

        let eq_class = &mut self.eq_classes[id];
        let hash = combined_hash(eq_class.representative(), &eq_class.child_ids, expr_arena);
        eq_class.members.remove(&node);

        if eq_class.members.is_empty() {
            self.eq_classes.remove(id);
            if let Ok(entry) = self
                .deduplication_map
                .find_entry(hash, |&other| other == id)
            {
                entry.remove();
            }
        }
    }

    /// Returns a representative node for `id`.
    pub fn representative(&self, id: CanonicalExprId) -> Node {
        self.eq_classes[id].representative()
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
        let Self {
            deduplication_map,
            eq_classes,
            ..
        } = self;

        let hash = combined_hash(node, &child_ids, expr_arena);
        let entry = deduplication_map.entry(
            hash,
            |&other| {
                let other = &eq_classes[other];
                child_ids == other.child_ids
                    && expr_arena
                        .get(node)
                        .is_expr_equal_shallow(expr_arena.get(other.representative()))
            },
            |&other| {
                let other = &eq_classes[other];
                combined_hash(other.representative(), &other.child_ids, expr_arena)
            },
        );

        match entry {
            Entry::Occupied(entry) => {
                let id = *entry.get();
                eq_classes[id].members.insert(node);
                id
            },
            Entry::Vacant(entry) => {
                let id = eq_classes.insert(CanonicalExprClass {
                    members: PlHashSet::from_iter([node]),
                    child_ids,
                });
                entry.insert(id);
                id
            },
        }
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
        self.get(expr.node()).hash(state);
        expr.output_name_inner().hash(state);
    }
}

/// Hashes the node itself, without descending into its children, combined with
/// the ids of those children.
fn combined_hash(node: Node, child_ids: &[CanonicalExprId], expr_arena: &Arena<AExpr>) -> u64 {
    let mut hasher = DefaultHasher::new();
    expr_arena.get(node).hash(&mut hasher);
    for child_id in child_ids {
        child_id.hash(&mut hasher);
    }
    hasher.finish()
}
