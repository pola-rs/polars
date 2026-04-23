use polars_core::error::PolarsResult;
use polars_core::prelude::PlHashSet;
use polars_utils::aliases::InitHashMaps;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use super::OptimizationRule;
use crate::prelude::{AExpr, IR};

pub struct FlattenMergeSortedRule {
    collected_inputs: Vec<Node>,
    optimized_nodes: PlHashSet<Node>,
    traversal_stack: Vec<Node>,
}

impl FlattenMergeSortedRule {
    pub fn new() -> Self {
        Self {
            collected_inputs: Vec::new(),
            optimized_nodes: PlHashSet::new(),
            traversal_stack: Vec::new(),
        }
    }
}

impl OptimizationRule for FlattenMergeSortedRule {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        _expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        let (key, maintain_order) = match lp_arena.get(node) {
            IR::MergeSorted {
                key,
                maintain_order,
                ..
            } => (key.clone(), *maintain_order),
            _ => return Ok(None),
        };
        if !self.optimized_nodes.insert(node) {
            return Ok(None);
        }

        self.collected_inputs.clear();
        collect_merge_sorted_inputs(
            node,
            &key,
            maintain_order,
            lp_arena,
            &mut self.collected_inputs,
            &mut self.traversal_stack,
        );

        if self.collected_inputs.len() <= 2 {
            return Ok(None);
        }

        Ok(Some(rebuild_merge_sorted_tree(
            &mut self.collected_inputs,
            key,
            maintain_order,
            lp_arena,
        )))
    }
}

fn collect_merge_sorted_inputs(
    root: Node,
    key: &PlSmallStr,
    maintain_order: bool,
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
                maintain_order: merge_maintain_order,
            } if merge_key == key && *merge_maintain_order == maintain_order => {
                traversal_stack.push(*input_right);
                traversal_stack.push(*input_left);
            },
            _ => out.push(node),
        }
    }
}

fn rebuild_merge_sorted_tree(
    inputs: &mut [Node],
    key: PlSmallStr,
    maintain_order: bool,
    lp_arena: &mut Arena<IR>,
) -> IR {
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
                maintain_order,
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
            maintain_order,
        }
    } else {
        unreachable!("rebuild_merge_sorted_tree requires at least 3 inputs")
    }
}
