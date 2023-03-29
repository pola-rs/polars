use polars_utils::arena::{Arena, Node};
use ALogicalPlan::*;

use super::OptimizationRule;
use crate::prelude::ALogicalPlan;

pub struct FlattenUnionRule {}

fn get_union_inputs(node: Node, lp_arena: &Arena<ALogicalPlan>) -> Option<&[Node]> {
    match lp_arena.get(node) {
        ALogicalPlan::Union { inputs, .. } => Some(inputs),
        _ => None,
    }
}

impl OptimizationRule for FlattenUnionRule {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut polars_utils::arena::Arena<ALogicalPlan>,
        _expr_arena: &mut polars_utils::arena::Arena<crate::prelude::AExpr>,
        node: polars_utils::arena::Node,
    ) -> Option<ALogicalPlan> {
        let lp = lp_arena.get(node);

        match lp {
            Union { inputs, options } => {
                if inputs
                    .iter()
                    .any(|node| matches!(lp_arena.get(*node), Union { .. }))
                {
                    let mut new_inputs = Vec::with_capacity(inputs.len() * 2);

                    for node in inputs {
                        match get_union_inputs(*node, lp_arena) {
                            Some(inp) => new_inputs.extend_from_slice(inp),
                            None => new_inputs.push(*node),
                        }
                    }
                    Some(Union {
                        inputs: new_inputs,
                        options: *options,
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}
