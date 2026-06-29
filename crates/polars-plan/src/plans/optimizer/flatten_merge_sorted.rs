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

        Ok(Some(build_merge_sorted_subtree(
            &self.collected_inputs,
            &key,
            maintain_order,
            lp_arena,
            &mut self.optimized_nodes,
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

fn build_merge_sorted_subtree(
    inputs: &[Node],
    key: &PlSmallStr,
    maintain_order: bool,
    lp_arena: &mut Arena<IR>,
    optimized_nodes: &mut PlHashSet<Node>,
) -> IR {
    debug_assert!(inputs.len() >= 2);

    let split = inputs.len().div_ceil(2);
    let input_left = build_merge_sorted_subtree_node(
        &inputs[..split],
        key,
        maintain_order,
        lp_arena,
        optimized_nodes,
    );
    let input_right = build_merge_sorted_subtree_node(
        &inputs[split..],
        key,
        maintain_order,
        lp_arena,
        optimized_nodes,
    );
    IR::MergeSorted {
        input_left,
        input_right,
        key: key.clone(),
        maintain_order,
    }
}

fn build_merge_sorted_subtree_node(
    inputs: &[Node],
    key: &PlSmallStr,
    maintain_order: bool,
    lp_arena: &mut Arena<IR>,
    optimized_nodes: &mut PlHashSet<Node>,
) -> Node {
    match inputs {
        [node] => *node,
        _ => {
            let subtree =
                build_merge_sorted_subtree(inputs, key, maintain_order, lp_arena, optimized_nodes);
            let node = lp_arena.add(subtree);
            optimized_nodes.insert(node);
            node
        },
    }
}
