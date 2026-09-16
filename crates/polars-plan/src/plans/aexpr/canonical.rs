use std::cell::RefCell;
use std::hash::{DefaultHasher, Hash, Hasher};

use hashbrown::HashTable;
use hashbrown::hash_table::Entry;
#[allow(clippy::disallowed_types)]
use polars_utils::aliases::PlHashMap;
use polars_utils::aliases::{InitHashMaps as _, PlIndexSet};
use polars_utils::arena::{Arena, Node};
use slotmap::SlotMap;

#[cfg(feature = "cse")]
use crate::plans::ExpressionHasher;
use crate::plans::aexpr::{
    is_inherently_nondeterministic_excluding_udfs_top_level,
    is_inherently_nondeterministic_top_level,
};
use crate::plans::{AExpr, ExprIR, ExpressionComparator};

slotmap::new_key_type! {
    /// Identifies an `AExpr` up to structural equality: two nodes get the same
    /// `CanonicalExprId` if and only if they represent the same expression.
    pub struct CanonicalExprId;
}

impl CanonicalExprId {
    /// Opaque integer representation, unique within a single [`CanonicalExprMap`].
    /// Only intended for generating unique names.
    pub fn as_u64(self) -> u64 {
        slotmap::Key::data(&self).as_ffi()
    }
}

/// Equivalence class of structurally equal expression nodes.
struct CanonicalExprClass {
    members: PlIndexSet<Node>,
    child_ids: Vec<CanonicalExprId>,
    /// Whether the subtree may produce a different value on each evaluation
    is_nondeterministic: bool,
    /// Like [`is_nondeterministic`], but excludes UDFs
    is_nondeterministic_excluding_udfs: bool,
}

impl CanonicalExprClass {
    /// An arbitrary member.
    fn representative(&self) -> Node {
        self.members[0]
    }
}

/// Assigns [`CanonicalExprId`]s to `AExpr` nodes.
pub struct CanonicalExprMap {
    deduplication_map: HashTable<CanonicalExprId>,
    #[allow(clippy::disallowed_types)] // We don't iterate over the cache.
    cache: PlHashMap<Node, CanonicalExprId>,
    eq_classes: SlotMap<CanonicalExprId, CanonicalExprClass>,
}

impl CanonicalExprMap {
    #[allow(clippy::disallowed_types)]
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
        eq_class.members.swap_remove(&node);

        if eq_class.members.is_empty() {
            let hash = combined_hash(node, &eq_class.child_ids, expr_arena);
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

    pub fn is_nondeterministic(&self, id: CanonicalExprId) -> bool {
        self.eq_classes[id].is_nondeterministic
    }

    pub fn is_nondeterministic_excluding_udfs(&self, id: CanonicalExprId) -> bool {
        self.eq_classes[id].is_nondeterministic_excluding_udfs
    }

    /// Returns the id of `node` if it was already resolved, without resolving it.
    pub fn cached_id(&self, node: Node) -> Option<CanonicalExprId> {
        self.cache.get(&node).copied()
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
                let child_ids = children.iter().map(|child| self.cache[child]).collect();
                let id = self.resolve_single(node, child_ids, expr_arena);
                self.cache.insert(node, id);
            } else {
                stack.push((node, true));
                children.clear();
                expr_arena.get(node).children_rev(&mut children);
                stack.extend(children.iter().map(|&child| (child, false)));
            }
        }

        self.cache[&node]
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
                let ae = expr_arena.get(node);
                let is_nondeterministic = is_inherently_nondeterministic_top_level(ae)
                    || child_ids
                        .iter()
                        .any(|&child| eq_classes[child].is_nondeterministic);
                let is_nondeterministic_excluding_udfs =
                    is_inherently_nondeterministic_excluding_udfs_top_level(ae)
                        || child_ids
                            .iter()
                            .any(|&child| eq_classes[child].is_nondeterministic_excluding_udfs);

                let id = eq_classes.insert(CanonicalExprClass {
                    members: PlIndexSet::from_iter([node]),
                    child_ids,
                    is_nondeterministic,
                    is_nondeterministic_excluding_udfs,
                });
                entry.insert(id);
                id
            },
        }
    }
}

impl Default for CanonicalExprMap {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CanonicalExprMapWithArena<'a> {
    map: RefCell<&'a mut CanonicalExprMap>,
    arena: &'a Arena<AExpr>,
}

impl<'a> CanonicalExprMapWithArena<'a> {
    pub fn new(map: &'a mut CanonicalExprMap, arena: &'a Arena<AExpr>) -> Self {
        Self {
            map: RefCell::new(map),
            arena,
        }
    }
}

impl<'a> ExpressionComparator for CanonicalExprMapWithArena<'a> {
    fn equals(&self, lhs: &ExprIR, rhs: &ExprIR) -> bool {
        let mut map = self.map.borrow_mut();
        map.resolve(lhs.node(), self.arena) == map.resolve(rhs.node(), self.arena)
            && lhs.output_name_inner() == rhs.output_name_inner()
    }
}

#[cfg(feature = "cse")]
impl<'a> ExpressionHasher for CanonicalExprMapWithArena<'a> {
    fn hash_expr<H: Hasher>(&self, expr: &ExprIR, state: &mut H) {
        self.map
            .borrow_mut()
            .resolve(expr.node(), self.arena)
            .hash(state);
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
