pub mod storage;

use std::ops::ControlFlow;

use polars_core::prelude::{InitHashMaps as _, PlHashMap};
use polars_utils::arena::{Arena, Node};
use polars_utils::unique_id::UniqueId;
use polars_utils::{UnitVec, unitvec};

use crate::plans::IR;
use crate::plans::optimizer::ir_traversal::storage::IRTraversalStorage;
use crate::traversal::tree_traversal::{GraphVisitOrder, TreeTraversalImpl};
use crate::traversal::visitor::{FnVisitors, NodeVisitor, SubtreeVisit};

#[derive(Debug)]
pub struct IRCacheNodeVisit<Key> {
    pub hits: usize,
    pub output_keys: UnitVec<Key>,
}

pub fn get_ir_cache_hits<Key>(
    root: Node,
    mut arena: &Arena<IR>,
    visit_stack: &mut Vec<Node>,
) -> PlHashMap<UniqueId, IRCacheNodeVisit<Key>> {
    let mut cache_out_edge_keys_map: PlHashMap<UniqueId, IRCacheNodeVisit<Key>> = PlHashMap::new();

    TreeTraversalImpl {
        storage: &mut arena,
        visit_stack,
        edges: &mut vec![()],
        persist_input_edge_idxs: None,
        graph_visit_order_fn: None,
        visitor: &mut FnVisitors::new(
            || (),
            |key, storage: &mut &Arena<IR>, _| {
                ControlFlow::Continue(if let IR::Cache { id, .. } = storage.get(key) {
                    use hashbrown::hash_map::Entry;

                    match cache_out_edge_keys_map.entry(*id) {
                        Entry::Occupied(mut e) => {
                            e.get_mut().hits += 1;
                            SubtreeVisit::Skip
                        },
                        Entry::Vacant(e) => {
                            e.insert(IRCacheNodeVisit {
                                hits: 1,
                                output_keys: unitvec![],
                            });
                            SubtreeVisit::Visit
                        },
                    }
                } else {
                    SubtreeVisit::Visit
                })
            },
            |_, _, _| ControlFlow::<()>::Continue(()),
        ),
    }
    .traverse_rec(root, 0, false)
    .continue_value()
    .unwrap();

    cache_out_edge_keys_map
}

pub fn ir_graph_traversal<'storage, Edge, BreakValue>(
    root: Node,
    visitor: &mut dyn NodeVisitor<
        Key = Node,
        Storage = IRTraversalStorage<'storage>,
        Edge = Edge,
        BreakValue = BreakValue,
    >,
    visit_stack: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    mut storage: IRTraversalStorage<'storage>,
) -> ControlFlow<BreakValue, Edge> {
    let mut cache_out_edge_keys_map = get_ir_cache_hits::<usize>(root, &storage, visit_stack);

    visit_stack.clear();
    let root_edge_idx = edges.len();
    edges.push(visitor.default_edge(root, None));
    let root_edge_deleted = visitor.is_deleted_edge(&edges[root_edge_idx]) == Some(true);

    TreeTraversalImpl {
        storage: &mut storage,
        visit_stack,
        edges,
        persist_input_edge_idxs: None,
        graph_visit_order_fn: Some(&mut |key, storage, edge_key_idx| {
            if let IR::Cache { id, .. } = storage.get(key) {
                let cache_node_visit = cache_out_edge_keys_map.get_mut(id).unwrap();

                if let Some(idx) = edge_key_idx {
                    cache_node_visit.output_keys.push(idx);
                }

                cache_node_visit.hits -= 1;

                if cache_node_visit.hits == 0 {
                    GraphVisitOrder::Visit {
                        output_keys: std::mem::take(&mut cache_node_visit.output_keys),
                    }
                } else {
                    GraphVisitOrder::HasUnvisitedOutputs
                }
            } else {
                GraphVisitOrder::Visit {
                    output_keys: if let Some(idx) = edge_key_idx {
                        unitvec![idx]
                    } else {
                        unitvec![]
                    },
                }
            }
        }),
        visitor,
    }
    .traverse_rec(root, root_edge_idx, root_edge_deleted)?;

    assert!(edges.len() >= root_edge_idx);
    edges.truncate(root_edge_idx + 1);

    ControlFlow::Continue(edges.pop().unwrap())
}
