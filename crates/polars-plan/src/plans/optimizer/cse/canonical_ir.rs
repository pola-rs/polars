use std::hash::{DefaultHasher, Hasher};

use hashbrown::HashTable;
use polars_core::prelude::{InitHashMaps as _, PlIndexMap};
use polars_utils::arena::{Arena, Node};

use crate::plans::aexpr::traverse_and_hash_aexpr;
use crate::plans::{AExpr, ArenaLpIter, ExprIR, ExpressionComparator, ExpressionHasher, IR};

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
///
/// Nodes must be canonicalized bottom-up: [`Self::get_or_assign`] takes the ids
/// that were already assigned to the inputs of the node.
pub struct CanonicalIRMap {
    deduplication_map: HashTable<CanonicalIREntry>,
    expr_cmp: HashExpressionCmp,
}

impl CanonicalIRMap {
    pub fn new(root: Node, lp_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Self {
        Self {
            deduplication_map: HashTable::new(),
            // Hash all exprs in advance, so that later comparison is immutable
            expr_cmp: HashExpressionCmp::new(root, lp_arena, expr_arena),
        }
    }

    /// Returns the id of `node`, given the ids of its inputs.
    pub fn get_or_assign(
        &mut self,
        node: Node,
        child_ids: Vec<CanonicalIRId>,
        lp_arena: &Arena<IR>,
    ) -> CanonicalIRId {
        let Self {
            deduplication_map,
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

/// Hashes the node itself, without descending into its inputs, combined with the
/// ids of those inputs.
fn combined_hash(
    node: Node,
    child_ids: &[CanonicalIRId],
    lp_arena: &Arena<IR>,
    expr_cmp: &HashExpressionCmp,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    lp_arena.get(node).shallow_hash(&mut hasher, expr_cmp);
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
