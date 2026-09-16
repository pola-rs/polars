use std::ops::ControlFlow;

use polars_utils::arena::{Arena, Node};
use polars_utils::itertools::Itertools as _;
use polars_utils::unique_id::UniqueId;

use crate::plans::{AExpr, IR, deep_clone_ae};
use crate::traversal::tree_traversal::tree_traversal;
use crate::traversal::visitor::{FnVisitors, SubtreeVisit};

/// Copies the `ir_node` and all nodes in the subtree rooted at `ir_node`, including expression nodes,
/// to new nodes in the arena. Cache nodes carrying `delete_id` are removed from the copy, i.e.
/// replaced by their input. Cache nodes with any other id are left intact.
pub(crate) fn deep_copy_ir_delete_cache_id(
    ir_node: Node,
    delete_id: UniqueId,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> Node {
    let cache_id = |ir_arena: &Arena<IR>, node: Node| match ir_arena.get(node) {
        IR::Cache { id, .. } => Some(*id),
        _ => None,
    };

    tree_traversal(
        ir_node,
        ir_arena,
        &mut vec![],
        &mut vec![],
        &mut FnVisitors::new(
            || ir_node,
            |node, ir_arena: &mut Arena<IR>, _| {
                // Don't descend into caches we keep; their subplan stays shared.
                ControlFlow::Continue(match cache_id(ir_arena, node) {
                    Some(id) if id != delete_id => SubtreeVisit::Skip,
                    _ => SubtreeVisit::Visit,
                })
            },
            |node, ir_arena: &mut Arena<IR>, edges| {
                let new_node = match cache_id(ir_arena, node) {
                    // Drop this cache: the copy uses its (copied) input directly.
                    Some(id) if id == delete_id => {
                        assert_eq!(edges.inputs().len(), 1);
                        assert_eq!(edges.outputs().len(), 1);
                        edges.inputs()[0]
                    },
                    // Keep this cache. Its subtree was skipped above, so the clone still points at
                    // the original input.
                    Some(_) => {
                        let ir_copy = ir_arena.get(node).clone();
                        ir_arena.add(ir_copy)
                    },
                    None => {
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
                    },
                };

                edges.outputs()[0] = new_node;

                ControlFlow::<()>::Continue(())
            },
        ),
    )
    .continue_value()
    .unwrap()
}
