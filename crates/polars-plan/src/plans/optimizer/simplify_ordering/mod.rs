pub(crate) mod expr;

use std::sync::Arc;

use polars_core::prelude::{InitHashMaps, PlHashMap};
use polars_utils::arena::{Arena, Node};
use polars_utils::unique_id::UniqueId;
use polars_utils::{UnitVec, unitvec};
use slotmap::{SlotMap, new_key_type};

use crate::plans::Sorted as SortingColumn;
use crate::prelude::{AExpr, IR};

enum Edge {
    Unordered,
    Ordered {
        sorting: Option<Arc<UnitVec<SortingColumn>>>,
    },
}

new_key_type! {
    struct EdgeKey;
}

type EdgesMap = SlotMap<EdgeKey, Option<Edge>>;

#[derive(Default, Debug)]
struct IRNodeEdges {
    in_edges: UnitVec<EdgeKey>,
    out_edges: UnitVec<EdgeKey>,
}

pub(crate) fn simplify_ir_ordering(
    roots: &[Node],
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) {
    let mut cache_hits: PlHashMap<UniqueId, usize> = PlHashMap::new();
    let mut num_nodes: usize = 0;

    let mut ir_nodes_stack = Vec::with_capacity(roots.len() + 8);
    ir_nodes_stack.extend_from_slice(roots);

    while let Some(ir_node) = ir_nodes_stack.pop() {
        let ir = ir_arena.get(ir_node);

        if let IR::Cache { id, .. } = ir {
            let _ = cache_hits.try_insert(*id, 0);
            *cache_hits.get_mut(id).unwrap() += 1;
        } else {
            num_nodes += 1;
        }

        ir.copy_inputs(&mut ir_nodes_stack);
    }

    num_nodes += cache_hits.len();

    let mut all_edges_map: SlotMap<EdgeKey, Option<Edge>> =
        SlotMap::with_capacity_and_key(num_nodes);
    let mut ir_node_to_edges_map: PlHashMap<Node, IRNodeEdges> =
        PlHashMap::with_capacity(num_nodes);

    ir_nodes_stack.reserve_exact(num_nodes);
    ir_nodes_stack.extend_from_slice(roots);

    for i in 0..(num_nodes + 1) {
        let Some(current_node) = ir_nodes_stack.get(i).copied() else {
            break;
        };

        assert!(i + 1 <= num_nodes);

        let ir = ir_arena.get(current_node);

        if let IR::Cache { id, .. } = ir {
            let hits = cache_hits.get_mut(id).unwrap();
            *hits -= 1;

            if *hits != 0 {
                debug_assert!(i < ir_nodes_stack.len());
                continue;
            }
        }

        let inputs_start_idx = ir_nodes_stack.len();
        ir_arena.get(current_node).copy_inputs(&mut ir_nodes_stack);
        let num_inputs = ir_nodes_stack.len() - inputs_start_idx;

        let mut current_node_in_edges =
            UnitVec::from_iter((0..num_inputs).map(|_| all_edges_map.insert(None)));

        for i in 0..num_inputs {
            let input_node = ir_nodes_stack[i + inputs_start_idx];
            let _ = ir_node_to_edges_map.try_insert(input_node, IRNodeEdges::default());
            let IRNodeEdges {
                out_edges: input_node_out_edges,
                ..
            } = ir_node_to_edges_map.get_mut(&input_node).unwrap();

            input_node_out_edges.push(current_node_in_edges[i])
        }

        let _ = ir_node_to_edges_map.try_insert(current_node, IRNodeEdges::default());
        let current_edges = ir_node_to_edges_map.get_mut(&current_node).unwrap();

        assert!(current_edges.in_edges.is_empty());
        current_edges.in_edges = current_node_in_edges;
    }

    // for node in ir_nodes_stack.iter().copied() {
    //     simplify_ir_node_ordering(
    //         node,
    //         &mut ir_node_to_edges_map,
    //         &mut all_edges_map,
    //         ir_arena,
    //         expr_arena,
    //     );
    // }

    // for node in ir_nodes_stack.drain(..).rev() {
    //     simplify_ir_node_ordering(
    //         node,
    //         &mut ir_node_to_edges_map,
    //         &mut all_edges_map,
    //         ir_arena,
    //         expr_arena,
    //     );
    // }

    // dbg!("OK");
}

// fn simplify_ir_node_ordering(
//     current_ir_node: Node,
//     ir_node_to_edges_map: &mut PlHashMap<Node, IRNodeEdges>,
//     all_edges_map: &mut EdgesMap,
//     ir_arena: &mut Arena<IR>,
//     expr_arena: &mut Arena<AExpr>,
// ) {
//     match ir_arena.get(current_ir_node) {
//         IR::Sink { input, payload } => {
//             let IRNodeEdges { in_edges, .. } =
//                 ir_node_to_edges_map.get_mut(&current_ir_node).unwrap();
//         },
//         IR::SinkMultiple { .. } | IR::Invalid => unreachable!(),
//         _ => {
//             dbg!();
//             unimplemented!()
//         },
//     }
// }
