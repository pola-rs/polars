use polars_core::error::PolarsResult;
use polars_utils::arena::{Arena, Node};

use super::OptimizationRule;
use crate::plans::aexpr::range_merge::merge_ranges_in_predicate;
use crate::prelude::{AExpr, IR};

pub struct RangeMergeRule {
    pub maintain_errors: bool,
}

impl OptimizationRule for RangeMergeRule {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        let IR::Filter { input, predicate } = lp_arena.get(node) else {
            return Ok(None);
        };
        let input = *input;
        let predicate_node = predicate.node();

        let Some(new_predicate_node) =
            merge_ranges_in_predicate(predicate_node, self.maintain_errors, expr_arena)
        else {
            return Ok(None);
        };
        // Today this is always a fresh `Literal(false)`, never the same node.
        // The check just stops the optimizer looping forever if that ever
        // changes.
        if new_predicate_node == predicate_node {
            return Ok(None);
        }

        // Clone only when we actually rewrite; the common path above returns
        // first.
        let mut new_predicate = predicate.clone();
        new_predicate.set_node(new_predicate_node);
        Ok(Some(IR::Filter {
            input,
            predicate: new_predicate,
        }))
    }
}
