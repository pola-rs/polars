use std::ops::ControlFlow;

use polars_utils::arena::{Arena, Node};
use polars_utils::itertools::Itertools as _;
use polars_utils::scratch_vec::ScratchVec;

use crate::plans::optimizer::ir_traversal::storage::IRTraversalStorageMut;
use crate::plans::{AExpr, IR};
use crate::traversal::tree_traversal::tree_traversal;
use crate::traversal::visitor::{FnVisitors, SubtreeVisit};

/// Copies the `ir_node` and all nodes in the subtree rooted at `ir_node`, including expression nodes,
/// to new nodes in the arena. The copied IR will have cache nodes removed.
pub(crate) fn deep_copy_ir_delete_caches(
    ir_node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    ir_nodes_scratch: &mut ScratchVec<Node>,
    ir_nodes_scratch2: &mut ScratchVec<Node>,
    ae_nodes_scratch: &mut ScratchVec<Node>,
    ae_nodes_scratch2: &mut ScratchVec<Node>,
) -> Node {
    let mut storage = IRTraversalStorageMut::new(ir_arena);

    tree_traversal(
        ir_node,
        &mut storage,
        ir_nodes_scratch.get(),
        ir_nodes_scratch2.get(),
        &mut FnVisitors::new(
            || ir_node,
            |_, _: &mut IRTraversalStorageMut<'_>, _| ControlFlow::Continue(SubtreeVisit::Visit),
            |node, ir_arena, edges| {
                let new_node = if let IR::Cache { .. } = ir_arena.get(node) {
                    assert_eq!(edges.inputs().len(), 1);
                    edges.inputs()[0]
                } else {
                    let mut ir_copy = ir_arena.get(node).clone();

                    for (orig_input, copied_input) in
                        ir_copy.inputs_mut().zip_eq(edges.inputs().iter().copied())
                    {
                        *orig_input = copied_input
                    }

                    for e in ir_copy.exprs_mut() {
                        e.set_node(deep_copy_ae(
                            e.node(),
                            expr_arena,
                            ae_nodes_scratch,
                            ae_nodes_scratch2,
                        ));
                    }

                    ir_arena.add(ir_copy)
                };

                edges.outputs()[0] = new_node;

                ControlFlow::<()>::Continue(())
            },
        ),
    )
    .continue_value()
    .unwrap()
}

/// Copies the `ae_node` and all nodes in the subtree rooted at `ae_node` to new nodes in the arena.
pub(crate) fn deep_copy_ae(
    ae_node: Node,
    expr_arena: &mut Arena<AExpr>,
    nodes_scratch: &mut ScratchVec<Node>,
    nodes_scratch2: &mut ScratchVec<Node>,
) -> Node {
    tree_traversal(
        ae_node,
        expr_arena,
        nodes_scratch.get(),
        nodes_scratch2.get(),
        &mut FnVisitors::new(
            || ae_node,
            |_, _: &mut Arena<AExpr>, _| ControlFlow::Continue(SubtreeVisit::Visit),
            |node, expr_arena, edges| {
                let mut ae_copy = expr_arena.get(node).clone();

                for (orig_input, copied_input) in ae_copy
                    .nodes_iter_mut()
                    .zip_eq(edges.inputs().iter().copied())
                {
                    *orig_input = copied_input
                }

                let new_node = expr_arena.add(ae_copy);
                edges.outputs()[0] = new_node;

                ControlFlow::<()>::Continue(())
            },
        ),
    )
    .continue_value()
    .unwrap()
}
