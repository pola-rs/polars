use std::ops::ControlFlow;

use polars_core::prelude::{InitHashMaps as _, PlIndexMap};
use polars_utils::arena::{Arena, Node};
use polars_utils::scratch_vec::ScratchVec;
use polars_utils::unique_id::UniqueId;

use super::{CanonicalIRId, CanonicalIRMap};
use crate::plans::{AExpr, IR};
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

    let canonical_ir_map = CanonicalIRMap::new();

    TreeTraversalImpl {
        storage: ir_arena,
        visit_stack: visit_stack.get(),
        edges: &mut edges,
        persist_input_edge_idxs: Some(&mut PersistInputEdgeIdxs::Build(
            &mut persisted_input_edge_idxs,
        )),
        graph_visit_order_fn: None,
        visitor: &mut IDGeneratorVisitor {
            canonical_ir_map,
            id_map: &mut id_map,
            expr_arena,
        },
    }
    .traverse_rec(root, 0, false)
    .continue_value()
    .unwrap();

    let mut inserted_cache = false;

    TreeTraversalImpl {
        storage: ir_arena,
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
    is_nondeterministic_excluding_udfs: bool,
}

impl IDState {
    fn new(is_nondeterministic_excluding_udfs: bool) -> Self {
        Self {
            hits: 1,
            replacement_ir: None,
            output_state_entry_idx: usize::MAX,
            is_nondeterministic_excluding_udfs,
        }
    }
}

struct IDGeneratorVisitor<'map, 'expr> {
    canonical_ir_map: CanonicalIRMap,
    id_map: &'map mut PlIndexMap<CanonicalIRId, IDState>,
    expr_arena: &'expr Arena<AExpr>,
}

impl NodeVisitor for IDGeneratorVisitor<'_, '_> {
    type Key = Node;
    type Storage = Arena<IR>;
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
        let id = self.canonical_ir_map.resolve(key, storage, self.expr_arena);

        use indexmap::map::Entry;

        let entry_idx = match self.id_map.entry(id) {
            Entry::Occupied(mut e) => {
                e.get_mut().hits += 1;
                e.index()
            },
            Entry::Vacant(e) => {
                let idx = e.index();

                e.insert(IDState::new(
                    self.canonical_ir_map.is_nondeterministic_excluding_udfs(id),
                ));

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

struct InsertCachesVisitor<'a> {
    id_map: &'a mut PlIndexMap<CanonicalIRId, IDState>,
    inserted_cache: &'a mut bool,
    insert_nested_caches: bool,
}

impl NodeVisitor for InsertCachesVisitor<'_> {
    type Key = Node;
    type Storage = Arena<IR>;
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

        // Cache the topmost deterministic node
        let should_cache = !curr_state.is_nondeterministic_excluding_udfs
            && if output_state.is_nondeterministic_excluding_udfs {
                curr_state.hits > 1
            } else {
                curr_state.hits > output_state.hits
            };

        if should_cache {
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
