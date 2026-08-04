use std::hash::Hasher;
use std::marker::PhantomData;
use std::ops::ControlFlow;

use polars_core::prelude::{InitHashMaps as _, PlIndexMap};
use polars_utils::arena::{Arena, Node};
use polars_utils::scratch_vec::ScratchVec;
use polars_utils::unique_id::UniqueId;

use super::interner::{DeduplicationId, Interner, ShallowNodeOps};
use crate::plans::aexpr::traverse_and_hash_aexpr;
use crate::plans::optimizer::ir_traversal::storage::IRTraversalStorage;
use crate::plans::{AExpr, ArenaLpIter, ExprIR, ExpressionComparator, ExpressionHasher, IR};
use crate::traversal::edge_provider::NodeEdgesProvider;
use crate::traversal::tree_traversal::{PersistInputEdgeIdxs, TreeTraversalImpl};
use crate::traversal::visitor::{NodeVisitor, SubtreeVisit};

/// Inserts `IR::Cache` on common subplans.
pub fn common_subplan_elimination(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
    insert_nested_caches: bool,
) -> bool {
    let mut visit_stack = ScratchVec::default();
    let mut edges = vec![usize::MAX]; // Indices into `id_map`
    let mut persisted_input_edge_idxs = vec![usize::MAX]; // For tree traversal
    let mut id_map = PlIndexMap::new();

    // Hash all exprs in advance, so that later comparison is immutable
    let expr_cmp = HashExpressionCmp::new(root, ir_arena, expr_arena);

    let mut storage = IRTraversalStorage { arena: ir_arena };

    TreeTraversalImpl {
        storage: &mut storage,
        visit_stack: visit_stack.get(),
        edges: &mut edges,
        persist_input_edge_idxs: Some(&mut PersistInputEdgeIdxs::Build(
            &mut persisted_input_edge_idxs,
        )),
        graph_visit_order_fn: None,
        visitor: &mut IDGeneratorVisitor {
            interner: Interner::new(),
            id_map: &mut id_map,
            phantom: PhantomData,
            expr_cmp,
        },
    }
    .traverse_rec(root, 0, false)
    .continue_value()
    .unwrap();

    let mut inserted_cache = false;

    TreeTraversalImpl {
        storage: &mut storage,
        visit_stack: visit_stack.get(),
        edges: &mut edges,
        persist_input_edge_idxs: Some(&mut PersistInputEdgeIdxs::Use(
            persisted_input_edge_idxs.as_slice(),
        )),
        graph_visit_order_fn: None,
        visitor: &mut InsertCachesVisitor {
            id_map: &mut id_map,
            inserted_cache: &mut inserted_cache,
            insert_nested_caches,
            phantom: PhantomData,
        },
    }
    .traverse_rec(root, 0, false)
    .continue_value()
    .unwrap();

    inserted_cache
}

#[derive(Debug)]
struct IDState {
    hits: usize,
    replacement_ir: Option<IR>,
    output_state_entry_idx: usize,
}

impl Default for IDState {
    fn default() -> Self {
        Self {
            hits: 1,
            replacement_ir: None,
            output_state_entry_idx: usize::MAX,
        }
    }
}

struct IDGeneratorVisitor<'map, 'arena> {
    interner: Interner<IR>,
    id_map: &'map mut PlIndexMap<DeduplicationId<IR>, IDState>,
    phantom: PhantomData<&'arena ()>,
    expr_cmp: HashExpressionCmp,
}

struct IrShallowOps<'a> {
    lp_arena: &'a Arena<IR>,
    expr_cmp: &'a HashExpressionCmp,
}

impl ShallowNodeOps for IrShallowOps<'_> {
    fn shallow_hash<H: Hasher>(&self, node: Node, state: &mut H) {
        self.lp_arena.get(node).shallow_hash(state, self.expr_cmp);
    }

    fn shallow_eq(&self, a: Node, b: Node) -> bool {
        self.lp_arena
            .get(a)
            .is_ir_equal_shallow(self.lp_arena.get(b), self.expr_cmp)
    }
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

impl<'map, 'arena> NodeVisitor for IDGeneratorVisitor<'map, 'arena> {
    type Key = Node;
    type Storage = IRTraversalStorage<'arena>;
    type Edge = usize;
    type BreakValue = ();

    fn default_edge(
        &mut self,
        _key: Self::Key,
        _parent_key_and_port: Option<(Self::Key, usize)>,
    ) -> Self::Edge {
        usize::MAX
    }

    fn pre_visit(
        &mut self,
        _key: Self::Key,
        _storage: &mut Self::Storage,
        _edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue, SubtreeVisit> {
        ControlFlow::Continue(SubtreeVisit::Visit)
    }

    fn post_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue> {
        let child_ids = edges
            .inputs()
            .iter()
            .map(|&i| *self.id_map.get_index(i).unwrap().0)
            .collect();

        let ops = IrShallowOps {
            lp_arena: &*storage.arena,
            expr_cmp: &self.expr_cmp,
        };
        let id = self.interner.get_or_assign(key, child_ids, &ops);

        use indexmap::map::Entry;

        let entry_idx = match self.id_map.entry(id) {
            Entry::Occupied(mut e) => {
                e.get_mut().hits += 1;
                e.index()
            },
            Entry::Vacant(e) => {
                let idx = e.index();

                e.insert(IDState::default());

                idx
            },
        };

        edges.outputs()[0] = entry_idx;

        for i in edges.inputs().iter().copied() {
            self.id_map
                .get_index_mut(i)
                .unwrap()
                .1
                .output_state_entry_idx = entry_idx
        }

        ControlFlow::Continue(())
    }
}

struct InsertCachesVisitor<'a, 'arena> {
    id_map: &'a mut PlIndexMap<DeduplicationId<IR>, IDState>,
    inserted_cache: &'a mut bool,
    insert_nested_caches: bool,
    phantom: PhantomData<&'arena ()>,
}

impl<'a, 'arena> NodeVisitor for InsertCachesVisitor<'a, 'arena> {
    type Key = Node;
    type Storage = IRTraversalStorage<'arena>;
    type Edge = usize;
    type BreakValue = ();

    fn default_edge(
        &mut self,
        _key: Self::Key,
        _parent_key_and_port: Option<(Self::Key, usize)>,
    ) -> Self::Edge {
        unreachable!()
    }

    fn pre_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue, SubtreeVisit> {
        let entry_idx_curr_node = edges.outputs()[0];
        let entry_idx_output_node = self
            .id_map
            .get_index(entry_idx_curr_node)
            .unwrap()
            .1
            .output_state_entry_idx;

        if entry_idx_output_node == usize::MAX {
            // We are at the root node
            assert_eq!(entry_idx_curr_node, self.id_map.len() - 1);
            return ControlFlow::Continue(SubtreeVisit::Visit);
        }

        let [(_, output_state), (_, curr_state)] = self
            .id_map
            .get_disjoint_indices_mut([entry_idx_output_node, entry_idx_curr_node])
            .unwrap();

        if curr_state.replacement_ir.is_some() {
            return ControlFlow::Continue(SubtreeVisit::Skip);
        }

        if curr_state.hits > output_state.hits {
            let replacement_ir = match storage.get(key) {
                ir @ IR::Cache { .. } => ir.clone(),
                _ => {
                    let ir = storage.take(key);
                    let new_key = storage.add(ir);

                    IR::Cache {
                        input: new_key,
                        id: UniqueId::new(),
                    }
                },
            };

            curr_state.replacement_ir = Some(replacement_ir);

            if !self.insert_nested_caches {
                return ControlFlow::Continue(SubtreeVisit::Skip);
            }
        }

        ControlFlow::Continue(SubtreeVisit::Visit)
    }

    fn post_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue> {
        let state = self.id_map.get_index(edges.outputs()[0]).unwrap().1;

        if let Some(replacement_ir) = state.replacement_ir.clone() {
            *storage.get_mut(key) = replacement_ir;
            *self.inserted_cache = true;
        }

        ControlFlow::Continue(())
    }
}
