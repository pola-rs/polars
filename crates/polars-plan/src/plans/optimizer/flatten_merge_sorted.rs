use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use crate::prelude::IR;

pub(super) fn optimize(root: Node, lp_arena: &mut Arena<IR>) {
    let mut pending = vec![root];
    let mut collected_inputs = Vec::new();
    let mut traversal_stack = Vec::new();

    while let Some(node) = pending.pop() {
        let rebuild_key = match lp_arena.get(node) {
            IR::MergeSorted { key, .. } => {
                collected_inputs.clear();
                collect_merge_sorted_inputs(
                    node,
                    key,
                    lp_arena,
                    &mut collected_inputs,
                    &mut traversal_stack,
                );
                pending.extend_from_slice(&collected_inputs);

                (collected_inputs.len() > 2).then(|| key.clone())
            },
            ir => {
                ir.copy_inputs(&mut pending);
                None
            },
        };

        if let Some(key) = rebuild_key {
            let rebuilt_ir = rebuild_merge_sorted_tree(&mut collected_inputs, key, lp_arena);
            lp_arena.replace(node, rebuilt_ir);
        }
    }
}

fn collect_merge_sorted_inputs(
    root: Node,
    key: &PlSmallStr,
    lp_arena: &Arena<IR>,
    out: &mut Vec<Node>,
    traversal_stack: &mut Vec<Node>,
) {
    traversal_stack.clear();
    traversal_stack.push(root);

    while let Some(node) = traversal_stack.pop() {
        match lp_arena.get(node) {
            IR::MergeSorted {
                input_left,
                input_right,
                key: merge_key,
            } if merge_key == key => {
                traversal_stack.push(*input_right);
                traversal_stack.push(*input_left);
            },
            _ => out.push(node),
        }
    }
}

fn rebuild_merge_sorted_tree(inputs: &mut [Node], key: PlSmallStr, lp_arena: &mut Arena<IR>) -> IR {
    debug_assert!(inputs.len() > 2);

    let mut len = inputs.len();

    while len > 2 {
        let pair_len = len & !1;
        let mut read = 0;
        let mut write = 0;

        while read < pair_len {
            inputs[write] = lp_arena.add(IR::MergeSorted {
                input_left: inputs[read],
                input_right: inputs[read + 1],
                key: key.clone(),
            });
            read += 2;
            write += 1;
        }

        if pair_len != len {
            inputs[write] = inputs[pair_len];
            write += 1;
        }

        len = write;
    }

    if let [input_left, input_right, ..] = inputs {
        IR::MergeSorted {
            input_left: *input_left,
            input_right: *input_right,
            key,
        }
    } else {
        unreachable!("rebuild_merge_sorted_tree requires at least 3 inputs")
    }
}
