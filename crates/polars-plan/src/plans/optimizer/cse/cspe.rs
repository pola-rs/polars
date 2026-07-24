use std::hash::{DefaultHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::ops::ControlFlow;

use hashbrown::HashTable;
use polars_core::prelude::{InitHashMaps as _, PlIndexMap};
use polars_utils::aliases::PlHashMap;
use polars_utils::arena::{Arena, Node};
use polars_utils::scratch_vec::ScratchVec;
use polars_utils::unique_id::UniqueId;

use crate::plans::aexpr::traverse_and_hash_aexpr;
use crate::plans::optimizer::ir_traversal::storage::IRTraversalStorage;
use crate::plans::visitor::hash::IRHashWrap;
use crate::plans::{AExpr, ExprIR, ExpressionComparator, IR};
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
    let mut deduplication_map = HashTable::default();
    let mut id_map = PlIndexMap::new();
    let mut storage = IRTraversalStorage {
        arena: ir_arena,
        skip_subtree: |ir| {
            match ir {
                // Don't visit all the files in a `scan *` operation.
                // Put an arbitrary limit to 20 files now.
                IR::Union {
                    options, inputs, ..
                } => options.from_partitioned_ds && inputs.len() > 20,
                _ => false,
            }
        },
    };

    TreeTraversalImpl {
        storage: &mut storage,
        visit_stack: visit_stack.get(),
        edges: &mut edges,
        persist_input_edge_idxs: Some(&mut PersistInputEdgeIdxs::Build(
            &mut persisted_input_edge_idxs,
        )),
        graph_visit_order_fn: None,
        visitor: &mut IDGeneratorVisitor {
            deduplication_map: &mut deduplication_map,
            id_map: &mut id_map,
            expr_arena,
            expr_cmp: HashExpressionCmp::new(),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
struct DeduplicationId(u32);

struct DeduplicationEntry {
    representative: Node,
    child_ids: Vec<DeduplicationId>,
    id: DeduplicationId,
}

struct IDGeneratorVisitor<'map, 'arena> {
    deduplication_map: &'map mut HashTable<DeduplicationEntry>,
    id_map: &'map mut PlIndexMap<DeduplicationId, IDState>,
    expr_arena: &'arena Arena<AExpr>,
    expr_cmp: HashExpressionCmp,
}

fn shallow_hasher<'a>(
    node: Node,
    child_ids: &[DeduplicationId],
    lp_arena: &'a Arena<IR>,
    expr_arena: &'a Arena<AExpr>,
) -> u64 {
    let mut hasher = DefaultHasher::new();

    IRHashWrap::new(node, lp_arena, expr_arena, true).hash(&mut hasher);
    for &child_id in child_ids {
        hasher.write_u32(child_id.0);
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

struct HashExpressionCmp {
    expr_hash_cache: PlHashMap<Node, [u8; 32]>,
}

impl HashExpressionCmp {
    fn new() -> Self {
        Self {
            expr_hash_cache: PlHashMap::new(),
        }
    }

    /// Blake3 digest of the expression, mirroring `ExprIR::traverse_and_hash`: the (cached)
    /// digest of the `AExpr` subtree, with the explicit alias mixed in. Keeping the alias in
    /// the digest keeps this comparator consistent with `shallow_hasher`/`IRHashWrap`, which
    /// also hash the alias.
    fn expr_hash(&mut self, expr: &ExprIR, expr_arena: &Arena<AExpr>) -> [u8; 32] {
        let tree_hash = *self.expr_hash_cache.entry(expr.node()).or_insert_with(|| {
            let mut hasher = Blake3Hasher::new();
            traverse_and_hash_aexpr(expr.node(), expr_arena, &mut hasher);
            hasher.finalize()
        });

        match expr.get_alias() {
            None => tree_hash,
            Some(alias) => {
                // `tree_hash` is fixed-width, so there's no boundary ambiguity with the alias.
                let mut hasher = Blake3Hasher::new();
                hasher.write(&tree_hash);
                hasher.write(alias.as_bytes());
                hasher.finalize()
            },
        }
    }
}

impl ExpressionComparator for HashExpressionCmp {
    fn equals(&mut self, lhs: &ExprIR, rhs: &ExprIR, expr_arena: &Arena<AExpr>) -> bool {
        self.expr_hash(lhs, expr_arena) == self.expr_hash(rhs, expr_arena)
    }
}

fn shallow_eq<'a>(
    lhs: Node,
    rhs: Node,
    lp_arena: &'a Arena<IR>,
    expr_arena: &'a Arena<AExpr>,
    expr_cmp: &mut HashExpressionCmp,
) -> bool {
    let lhs = lp_arena.get(lhs);
    let rhs = lp_arena.get(rhs);

    lhs.is_ir_equal_shallow(rhs, expr_arena, expr_cmp)
}

fn get_deduplication_id<'a>(
    deduplication_map: &'a mut HashTable<DeduplicationEntry>,
    node: Node,
    child_ids: Vec<DeduplicationId>,
    lp_arena: &'a Arena<IR>,
    expr_arena: &'a Arena<AExpr>,
    expr_cmp: &mut HashExpressionCmp,
) -> DeduplicationId {
    let shallow_hash = shallow_hasher(node, &child_ids, lp_arena, expr_arena);

    let next_id: DeduplicationId = DeduplicationId(1 + deduplication_map.len() as u32);
    deduplication_map
        .entry(
            shallow_hash,
            |other| {
                shallow_eq(node, other.representative, lp_arena, expr_arena, expr_cmp)
                    && child_ids == other.child_ids
            },
            |other| shallow_hasher(other.representative, &child_ids, lp_arena, expr_arena),
        )
        .or_insert(DeduplicationEntry {
            representative: node,
            child_ids,
            id: next_id,
        })
        .get()
        .id
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
        let id = get_deduplication_id(
            self.deduplication_map,
            key,
            child_ids,
            storage.arena,
            self.expr_arena,
            &mut self.expr_cmp,
        );

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
    id_map: &'a mut PlIndexMap<DeduplicationId, IDState>,
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
