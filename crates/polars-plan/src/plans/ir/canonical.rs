use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::ControlFlow;

use hashbrown::HashTable;
use polars_core::prelude::{InitHashMaps as _, PlIndexMap};
use polars_utils::arena::{Arena, Node};

use crate::plans::{AExpr, CanonicalExprMap, CanonicalExprMapWithArena, IR};
use crate::traversal::edge_provider::NodeEdgesProvider;
use crate::traversal::tree_traversal::tree_traversal;
use crate::traversal::visitor::{NodeVisitor, SubtreeVisit};

/// Identifies an `IR` subplan up to structural equality: two nodes get the same
/// `CanonicalIRId` if and only if they represent the same plan.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CanonicalIRId(u32);

impl CanonicalIRId {
    fn index(self) -> usize {
        self.0 as usize
    }
}

struct CanonicalIRClass {
    representative: Node,
    child_ids: Vec<CanonicalIRId>,
    is_nondeterministic_excluding_udfs: bool,
}

/// Assigns [`CanonicalIRId`]s to `IR` nodes.
pub struct CanonicalIRMap {
    deduplication_map: HashTable<CanonicalIRId>,
    cache: PlIndexMap<Node, CanonicalIRId>,
    /// Indexed by [`CanonicalIRId`], which are handed out densely.
    eq_classes: Vec<CanonicalIRClass>,
    expr_map: CanonicalExprMap,
}

impl CanonicalIRMap {
    pub fn new() -> Self {
        Self {
            deduplication_map: HashTable::new(),
            cache: PlIndexMap::new(),
            eq_classes: Vec::new(),
            expr_map: CanonicalExprMap::new(),
        }
    }

    pub fn is_nondeterministic_excluding_udfs(&self, id: CanonicalIRId) -> bool {
        self.eq_classes[id.index()].is_nondeterministic_excluding_udfs
    }

    /// Returns the id of `node`, resolving its children recursively
    pub fn resolve(
        &mut self,
        node: Node,
        lp_arena: &Arena<IR>,
        expr_arena: &Arena<AExpr>,
    ) -> CanonicalIRId {
        tree_traversal(
            node,
            &mut &*lp_arena,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut ResolveVisitor {
                map: self,
                expr_arena,
            },
        )
        .continue_value()
        .unwrap()
        .unwrap()
    }

    /// Returns the id of `node`, given the ids of its inputs.
    fn resolve_single(
        &mut self,
        node: Node,
        child_ids: Vec<CanonicalIRId>,
        lp_arena: &Arena<IR>,
        expr_arena: &Arena<AExpr>,
    ) -> CanonicalIRId {
        let exprs_are_nondeterministic = lp_arena.get(node).exprs().any(|expr| {
            let expr_id = self.expr_map.resolve(expr.node(), expr_arena);
            self.expr_map.is_nondeterministic_excluding_udfs(expr_id)
        });

        let Self {
            deduplication_map,
            cache,
            eq_classes,
            expr_map,
        } = self;

        let expr_cmp = CanonicalExprMapWithArena::new(expr_map, expr_arena);
        let hash = combined_hash(node, &child_ids, lp_arena, &expr_cmp);
        let entry = deduplication_map.entry(
            hash,
            |&other| {
                let other = &eq_classes[other.index()];
                child_ids == other.child_ids
                    && lp_arena
                        .get(node)
                        .is_ir_equal_shallow(lp_arena.get(other.representative), &expr_cmp)
            },
            |&other| {
                let other = &eq_classes[other.index()];
                combined_hash(other.representative, &other.child_ids, lp_arena, &expr_cmp)
            },
        );

        let id = *entry
            .or_insert_with(|| {
                let is_nondeterministic_excluding_udfs = exprs_are_nondeterministic
                    || child_ids
                        .iter()
                        .any(|&child| eq_classes[child.index()].is_nondeterministic_excluding_udfs);

                let id = CanonicalIRId(eq_classes.len() as u32);
                eq_classes.push(CanonicalIRClass {
                    representative: node,
                    child_ids,
                    is_nondeterministic_excluding_udfs,
                });
                id
            })
            .get();

        cache.insert(node, id);
        id
    }
}

impl Default for CanonicalIRMap {
    fn default() -> Self {
        Self::new()
    }
}

struct ResolveVisitor<'a, 'arena> {
    map: &'a mut CanonicalIRMap,
    expr_arena: &'arena Arena<AExpr>,
}

impl<'arena> NodeVisitor for ResolveVisitor<'_, 'arena> {
    type Key = Node;
    type Storage = &'arena Arena<IR>;
    type Edge = Option<CanonicalIRId>;
    type BreakValue = ();

    fn default_edge(
        &mut self,
        _key: Self::Key,
        _parent_key_and_port: Option<(Self::Key, usize)>,
    ) -> Self::Edge {
        None
    }

    fn pre_visit(
        &mut self,
        key: Self::Key,
        _storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue, SubtreeVisit> {
        // The inputs of an already resolved node do not have to be visited.
        ControlFlow::Continue(match self.map.cache.get(&key) {
            Some(&id) => {
                edges.outputs()[0] = Some(id);
                SubtreeVisit::Skip
            },
            None => SubtreeVisit::Visit,
        })
    }

    fn post_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue> {
        // `post_visit` is still called even with `SubtreeVisit::Skip`
        if edges.outputs()[0].is_some() {
            return ControlFlow::Continue(());
        }

        let child_ids = edges
            .inputs()
            .iter()
            .map(|id| id.expect("input was not resolved"))
            .collect();

        let id = self
            .map
            .resolve_single(key, child_ids, storage, self.expr_arena);

        edges.outputs()[0] = Some(id);

        ControlFlow::Continue(())
    }
}

/// Hashes the node itself, without descending into its inputs, combined with the
/// ids of those inputs.
fn combined_hash(
    node: Node,
    child_ids: &[CanonicalIRId],
    lp_arena: &Arena<IR>,
    expr_cmp: &CanonicalExprMapWithArena,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    lp_arena
        .get(node)
        .hash_excluding_inputs(&mut hasher, expr_cmp);
    for child_id in child_ids {
        child_id.hash(&mut hasher);
    }
    hasher.finish()
}
