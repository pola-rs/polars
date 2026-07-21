use std::hash::{DefaultHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::ops::ControlFlow;

use hashbrown::HashTable;
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
}

fn shallow_hasher<'a>(
    node: Node,
    childs_ids: &[DeduplicationId],
    lp_arena: &'a Arena<IR>,
    expr_arena: &'a Arena<AExpr>,
) -> u64 {
    let mut hasher = DefaultHasher::new();

    IRHashWrap::new(node, lp_arena, expr_arena, true).hash(&mut hasher);
    for &child_id in childs_ids {
        hasher.write_u32(child_id.0);
    }

    hasher.finish()
}

fn shallow_eq<'a>(
    lhs: Node,
    rhs: Node,
    lp_arena: &'a Arena<IR>,
    expr_arena: &'a Arena<AExpr>,
) -> bool {
    let lhs = lp_arena.get(lhs);
    let rhs = lp_arena.get(rhs);

    lhs.is_ir_equal_shallow(rhs, expr_arena)
}

fn get_deduplication_id<'a>(
    deduplication_map: &'a mut HashTable<DeduplicationEntry>,
    node: Node,
    child_ids: Vec<DeduplicationId>,
    lp_arena: &'a Arena<IR>,
    expr_arena: &'a Arena<AExpr>,
) -> DeduplicationId {
    let shallow_hash = shallow_hasher(node, &child_ids, lp_arena, expr_arena);

    let next_id: DeduplicationId = DeduplicationId(1 + deduplication_map.len() as u32);
    deduplication_map
        .entry(
            shallow_hash,
            |other| {
                shallow_eq(node, other.representative, lp_arena, expr_arena)
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
            &self.expr_arena,
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
