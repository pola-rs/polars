use polars_core::prelude::{InitHashMaps, PlHashMap};
use polars_utils::UnitVec;
use polars_utils::arena::{Arena, Node};
use polars_utils::array::{array_concat, array_split};
use polars_utils::unique_id::UniqueId;
use slotmap::SlotMap;

use crate::plans::ir_traversal::ir_node_key::IRNodeKey;
use crate::prelude::IR;

#[derive(Default, Debug)]
pub struct IRNodeEdgeKeys<EdgeKey> {
    pub in_edges: UnitVec<EdgeKey>,
    pub out_edges: UnitVec<EdgeKey>,
    pub out_nodes: UnitVec<Node>,
}

/// Cache nodes that share a cache ID.
struct CacheNodes {
    nodes: Vec<Node>,
    hits: usize,
}

#[derive(Default)]
pub struct CacheNodeUpdater {
    inner: PlHashMap<UniqueId, CacheNodes>,
}

impl CacheNodeUpdater {
    pub fn update_cache_nodes(self, ir_arena: &mut Arena<IR>) {
        for (_, CacheNodes { nodes, hits: _ }) in self.inner {
            let IR::Cache { input, .. } = ir_arena.get(nodes[0]) else {
                unreachable!()
            };
            let updated_input = *input;

            for node in nodes.into_iter().skip(1) {
                let IR::Cache { input, .. } = ir_arena.get_mut(node) else {
                    unreachable!()
                };
                *input = updated_input;
            }
        }
    }
}

/// Builds an IR traversal graph where caches are visited only after all of their consumers are
/// visited.
#[expect(clippy::type_complexity)]
pub fn build_ir_traversal_graph<EdgeKey, Edge>(
    roots: &[Node],
    ir_arena: &mut Arena<IR>,
) -> (
    Vec<Node>,                                     // Nodes in sink->source traversal order
    PlHashMap<IRNodeKey, IRNodeEdgeKeys<EdgeKey>>, // Edge keys for each node
    SlotMap<EdgeKey, Edge>,                        // Edges slotmap
    CacheNodeUpdater,                              // All arena nodes that use this cache ID.
)
where
    EdgeKey: slotmap::Key,
    Edge: Default,
{
    let mut cache_track: PlHashMap<UniqueId, CacheNodes> = PlHashMap::new();
    let mut num_nodes: usize = 0;

    let mut ir_nodes_stack = Vec::with_capacity(roots.len() + 8);
    ir_nodes_stack.extend_from_slice(roots);

    while let Some(ir_node) = ir_nodes_stack.pop() {
        let ir = ir_arena.get(ir_node);

        if let IR::Cache { id, .. } = ir {
            use hashbrown::hash_map::Entry;

            match cache_track.entry(*id) {
                Entry::Occupied(mut v) => {
                    let tracker = v.get_mut();
                    tracker.hits += 1;
                    tracker.nodes.push(ir_node);
                    continue;
                },
                Entry::Vacant(v) => {
                    v.insert(CacheNodes {
                        nodes: vec![ir_node],
                        hits: 1,
                    });
                },
            }
        }

        num_nodes += 1;
        ir.copy_inputs(&mut ir_nodes_stack);
    }

    num_nodes += cache_track.len();

    let mut all_edges_map: SlotMap<EdgeKey, Edge> = SlotMap::with_capacity_and_key(num_nodes);
    let mut ir_node_to_edges_map: PlHashMap<IRNodeKey, IRNodeEdgeKeys<EdgeKey>> =
        PlHashMap::with_capacity(num_nodes);

    ir_nodes_stack.reserve_exact(num_nodes);
    ir_nodes_stack.extend_from_slice(roots);

    let iterations: usize = num_nodes + cache_track.values().map(|v| v.hits - 1).sum::<usize>();

    for i in 0..usize::MAX {
        let Some(mut current_node) = ir_nodes_stack.get(i).copied() else {
            break;
        };

        debug_assert!(i < iterations);

        let ir = ir_arena.get(current_node);

        if let IR::Cache { id, .. } = ir {
            let tracker = cache_track.get_mut(id).unwrap();
            tracker.hits -= 1;

            if tracker.hits != 0 {
                debug_assert!(i < ir_nodes_stack.len());
                continue;
            }

            current_node = tracker.nodes[0]
        }

        let inputs_start_idx = ir_nodes_stack.len();
        ir_arena.get(current_node).copy_inputs(&mut ir_nodes_stack);
        let num_inputs = ir_nodes_stack.len() - inputs_start_idx;

        let current_node_in_edges =
            UnitVec::from_iter((0..num_inputs).map(|_| all_edges_map.insert(Edge::default())));

        for i in 0..num_inputs {
            let input_node = ir_nodes_stack[i + inputs_start_idx];
            let input_node_key = IRNodeKey::new(input_node, ir_arena);
            let _ = ir_node_to_edges_map.try_insert(input_node_key, IRNodeEdgeKeys::default());
            let IRNodeEdgeKeys {
                out_edges: input_node_out_edges,
                out_nodes: input_node_out_nodes,
                ..
            } = ir_node_to_edges_map.get_mut(&input_node_key).unwrap();

            input_node_out_edges.push(current_node_in_edges[i]);
            input_node_out_nodes.push(current_node);
        }

        let current_node_key = IRNodeKey::new(current_node, ir_arena);

        let _ = ir_node_to_edges_map.try_insert(current_node_key, IRNodeEdgeKeys::default());
        let current_edges = ir_node_to_edges_map.get_mut(&current_node_key).unwrap();

        assert!(current_edges.in_edges.is_empty());
        current_edges.in_edges = current_node_in_edges;
    }

    (
        ir_nodes_stack,
        ir_node_to_edges_map,
        all_edges_map,
        CacheNodeUpdater { inner: cache_track },
    )
}

pub fn unpack_edges_mut<
    'a,
    EdgeKey: slotmap::Key,
    Edge,
    const NUM_INPUTS: usize,
    const NUM_OUTPUTS: usize,
    // Workaround for generic_const_exprs, have the caller pass in `NUM_INPUTS + NUM_OUTPUTS`
    const TOTAL_EDGES: usize,
>(
    node_edge_keys: &IRNodeEdgeKeys<EdgeKey>,
    edges_map: &'a mut SlotMap<EdgeKey, Edge>,
) -> Option<([&'a mut Edge; NUM_INPUTS], [&'a mut Edge; NUM_OUTPUTS])> {
    const {
        assert!(NUM_INPUTS + NUM_OUTPUTS == TOTAL_EDGES);
    }

    let in_: [EdgeKey; NUM_INPUTS] = node_edge_keys.in_edges.as_slice().try_into().ok()?;
    let out: [EdgeKey; NUM_OUTPUTS] = node_edge_keys.out_edges.as_slice().try_into().ok()?;

    let combined: [EdgeKey; TOTAL_EDGES] = array_concat(in_, out);
    let combined: [&mut Edge; TOTAL_EDGES] = edges_map.get_disjoint_mut(combined).unwrap();

    Some(array_split(combined))
}
