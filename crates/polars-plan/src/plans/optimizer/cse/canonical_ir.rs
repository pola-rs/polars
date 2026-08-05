use std::hash::{DefaultHasher, Hasher};
use std::marker::PhantomData;
use std::ops::ControlFlow;

use hashbrown::HashTable;
use polars_core::prelude::{InitHashMaps as _, PlIndexMap};
use polars_utils::arena::{Arena, Node};

use super::canonical_expr::CanonicalExprMap;
use crate::plans::{AExpr, ArenaLpIter, IR};
use crate::traversal::edge_provider::NodeEdgesProvider;
use crate::traversal::tree_traversal::tree_traversal;
use crate::traversal::visitor::{NodeVisitor, SubtreeVisit};

/// Identifies an `IR` subplan up to structural equality: two nodes get the same
/// `CanonicalIRId` if and only if they represent the same plan.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CanonicalIRId(u32);

impl CanonicalIRId {
    fn as_u32(self) -> u32 {
        self.0
    }
}

struct CanonicalIREntry {
    representative: Node,
    child_ids: Vec<CanonicalIRId>,
    id: CanonicalIRId,
}

/// Assigns [`CanonicalIRId`]s to `IR` nodes.
pub struct CanonicalIRMap {
    deduplication_map: HashTable<CanonicalIREntry>,
    cache: PlIndexMap<Node, CanonicalIRId>,
    expr_cmp: CanonicalExprMap,
}

impl CanonicalIRMap {
    pub fn new(root: Node, lp_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Self {
        // Resolve all expressions in advance so that IR comparison can stay immutable.
        let mut expr_cmp = CanonicalExprMap::new();
        for (_, ir) in lp_arena.iter(root) {
            for expr in ir.exprs() {
                expr_cmp.resolve(expr.node(), expr_arena);
            }
        }

        Self {
            deduplication_map: HashTable::new(),
            cache: PlIndexMap::new(),
            expr_cmp,
        }
    }

    /// Returns the id of `node`, resolving its children recursively
    pub fn resolve(&mut self, node: Node, lp_arena: &Arena<IR>) -> CanonicalIRId {
        if let Some(id) = self.cache.get(&node) {
            return *id;
        }

        tree_traversal(
            node,
            &mut &*lp_arena,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut ResolveVisitor {
                map: self,
                phantom: PhantomData,
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
    ) -> CanonicalIRId {
        let Self {
            deduplication_map,
            cache: _,
            expr_cmp,
        } = self;

        let hash = combined_hash(node, &child_ids, lp_arena, expr_cmp);
        let next_id = CanonicalIRId(1 + deduplication_map.len() as u32);
        deduplication_map
            .entry(
                hash,
                |other| {
                    child_ids == other.child_ids
                        && lp_arena
                            .get(node)
                            .is_ir_equal_shallow(lp_arena.get(other.representative), expr_cmp)
                },
                |other| combined_hash(other.representative, &other.child_ids, lp_arena, expr_cmp),
            )
            .or_insert(CanonicalIREntry {
                representative: node,
                child_ids,
                id: next_id,
            })
            .get()
            .id
    }
}

struct ResolveVisitor<'a, 'arena> {
    map: &'a mut CanonicalIRMap,
    phantom: PhantomData<&'arena Arena<IR>>,
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
        _edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue, SubtreeVisit> {
        // The inputs of an already resolved node do not have to be visited.
        ControlFlow::Continue(if self.map.cache.contains_key(&key) {
            SubtreeVisit::Skip
        } else {
            SubtreeVisit::Visit
        })
    }

    fn post_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue> {
        let id = match self.map.cache.get(&key) {
            Some(id) => *id,
            None => {
                let child_ids = edges
                    .inputs()
                    .iter()
                    .map(|id| id.expect("input was not resolved"))
                    .collect();

                let id = self.map.resolve_single(key, child_ids, storage);
                self.map.cache.insert(key, id);
                id
            },
        };

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
    expr_cmp: &CanonicalExprMap,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    lp_arena
        .get(node)
        .hash_excluding_inputs(&mut hasher, expr_cmp);
    for child_id in child_ids {
        hasher.write_u32(child_id.as_u32());
    }
    hasher.finish()
}
