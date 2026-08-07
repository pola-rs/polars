use std::ops::ControlFlow;

use polars_utils::arena::{Arena, Node};
use polars_utils::itertools::Itertools as _;

use crate::plans::{AExpr, IR, deep_clone_ae};
use crate::traversal::tree_traversal::tree_traversal;
use crate::traversal::visitor::{FnVisitors, SubtreeVisit};

/// Copies the `ir_node` and all nodes in the subtree rooted at `ir_node`, including expression nodes,
/// to new nodes in the arena. The copied IR will have cache nodes removed.
pub(crate) fn deep_copy_ir_delete_caches(
    ir_node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> Node {
    tree_traversal(
        ir_node,
        ir_arena,
        &mut vec![],
        &mut vec![],
        &mut FnVisitors::new(
            || ir_node,
            |_, _: &mut Arena<IR>, _| ControlFlow::Continue(SubtreeVisit::Visit),
            |node, ir_arena, edges| {
                let new_node = if let IR::Cache { .. } = ir_arena.get(node) {
                    assert_eq!(edges.inputs().len(), 1);
                    assert_eq!(edges.outputs().len(), 1);
                    edges.inputs()[0]
                } else {
                    let mut ir_copy = ir_arena.get(node).clone();

                    for (orig_input, copied_input) in
                        ir_copy.inputs_mut().zip_eq(edges.inputs().iter().copied())
                    {
                        *orig_input = copied_input
                    }

                    for e in ir_copy.exprs_mut() {
                        e.set_node(deep_clone_ae(e.node(), expr_arena));
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
