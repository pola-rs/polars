use std::hash::{DefaultHasher, Hasher};
use std::marker::PhantomData;
use std::ops::ControlFlow;

use hashbrown::HashTable;
use polars_core::prelude::{InitHashMaps as _, PlIndexMap};
use polars_utils::arena::{Arena, Node};

use crate::plans::aexpr::traverse_and_hash_aexpr;
use crate::plans::{AExpr, ArenaLpIter, ExprIR, ExpressionComparator, ExpressionHasher, IR};
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
    expr_cmp: HashExpressionCmp,
}

impl CanonicalIRMap {
    pub fn new(root: Node, lp_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Self {
        Self {
            deduplication_map: HashTable::new(),
            cache: PlIndexMap::new(),
            // Hash all exprs in advance, so that later comparison is immutable
            expr_cmp: HashExpressionCmp::new(root, lp_arena, expr_arena),
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
    expr_cmp: &HashExpressionCmp,
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

struct Blake3Hasher {
    hasher: blake3::Hasher,
}

impl Blake3Hasher {
    fn new() -> Self {
        Self {
            hasher: blake3::Hasher::new(),
        }
    }

    fn finalize(self) -> [u8; 32] {
        self.hasher.finalize().into()
    }
}

impl Hasher for Blake3Hasher {
    fn finish(&self) -> u64 {
        0
    }

    fn write(&mut self, bytes: &[u8]) {
        self.hasher.update(bytes);
    }
}

/// Compares expressions by a precomputed hash of their whole subtree, so that
/// comparing two `IR` nodes does not have to descend into their expressions.
struct HashExpressionCmp {
    expr_hashes: PlIndexMap<Node, [u8; 32]>,
}

impl HashExpressionCmp {
    fn new(root: Node, lp_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Self {
        let mut expr_hashes = PlIndexMap::new();
        for (_, ir) in lp_arena.iter(root) {
            for e in ir.exprs() {
                expr_hashes.entry(e.node()).or_insert_with(|| {
                    let mut hasher = Blake3Hasher::new();
                    traverse_and_hash_aexpr(e.node(), expr_arena, &mut hasher);
                    hasher.finalize()
                });
            }
        }
        Self { expr_hashes }
    }

    // Multiple ExprIRs can reference the same Node, but have different names.
    // The alias is included in the IRHashWrap hasher, so we need to include it here as well.
    fn hash_with_alias(&self, expr: &ExprIR) -> [u8; 32] {
        let tree_hash = self.expr_hashes[&expr.node()];
        match expr.get_alias() {
            None => tree_hash,
            Some(alias) => {
                let mut hasher = Blake3Hasher::new();
                hasher.write(&tree_hash);
                hasher.write(alias.as_bytes());
                hasher.finalize()
            },
        }
    }
}

impl ExpressionComparator for HashExpressionCmp {
    fn equals(&self, lhs: &ExprIR, rhs: &ExprIR) -> bool {
        self.hash_with_alias(lhs) == self.hash_with_alias(rhs)
    }
}

impl ExpressionHasher for HashExpressionCmp {
    fn hash_expr<H: Hasher>(&self, expr: &ExprIR, state: &mut H) {
        state.write(&self.hash_with_alias(expr));
    }
}
