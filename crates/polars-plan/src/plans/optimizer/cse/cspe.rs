use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::ControlFlow;

use polars_core::prelude::{InitHashMaps as _, PlIndexMap};
use polars_utils::arena::{Arena, Node};
use polars_utils::scratch_vec::ScratchVec;
use polars_utils::unique_id::UniqueId;

use crate::plans::optimizer::ir_traversal::storage::IRTraversalStorage;
use crate::plans::visitor::hash::IRHashWrap;
use crate::plans::{AExpr, IR};
use crate::traversal::edge_provider::NodeEdgesProvider;
use crate::traversal::tree_traversal::{PersistInputEdgeIdxs, TreeTraversalImpl};
use crate::traversal::visitor::{NodeVisitor, SubtreeVisit};

/// Inserts `IR::Cache` on common subplans.
pub fn common_subplan_elimination(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> bool {
    let mut visit_stack = ScratchVec::default();
    let mut edges = vec![usize::MAX]; // Indices into `id_map`
    let mut persisted_input_edge_idxs = vec![usize::MAX]; // For tree traversal
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
            id_map: &mut id_map,
            expr_arena,
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
            phantom: PhantomData,
        },
    }
    .traverse_rec(root, 0, false)
    .continue_value()
    .unwrap();

    inserted_cache
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
    id_map: &'map mut PlIndexMap<[u8; 32], IDState>,
    expr_arena: &'arena Arena<AExpr>,
}

impl<'map, 'arena> NodeVisitor for IDGeneratorVisitor<'map, 'arena> {
    type Key = Node;
    type Edge = usize;
    type Storage = IRTraversalStorage<'arena>;
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
        let ir = storage.get(key);

        let mut hasher = Blake3Hasher::new();

        hasher.write_usize(if storage.skip_subtree(ir) {
            // Subtree nodes were not pushed for traversal due to e.g. too many
            // union input nodes. We hash the memory address of this &IR instead.
            ir as *const IR as usize
        } else {
            0
        });

        IRHashWrap::new(key, storage, self.expr_arena, true).hash(&mut hasher);

        for entry_idx in edges.inputs().iter().copied() {
            let input_hash: &[u8; 32] = self.id_map.get_index(entry_idx).unwrap().0;
            hasher.write(input_hash);
        }

        let id: [u8; 32] = hasher.finalize();

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
    id_map: &'a mut PlIndexMap<[u8; 32], IDState>,
    inserted_cache: &'a mut bool,
    phantom: PhantomData<&'arena ()>,
}

impl<'a, 'arena> NodeVisitor for InsertCachesVisitor<'a, 'arena> {
    type Key = Node;
    type Edge = usize;
    type Storage = IRTraversalStorage<'arena>;
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

            // TODO: Remove this to enabled nested CSPE
            return ControlFlow::Continue(SubtreeVisit::Skip);
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
