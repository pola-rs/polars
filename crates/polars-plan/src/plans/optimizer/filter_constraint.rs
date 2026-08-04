use polars_core::error::PolarsResult;
use polars_utils::arena::{Arena, Node};

use super::OptimizationRule;
use crate::plans::aexpr::filter_constraint::merge_filter_constraints;
use crate::prelude::{AExpr, IR};

pub struct FilterConstraintRule {
    pub maintain_errors: bool,
}

impl OptimizationRule for FilterConstraintRule {
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
            merge_filter_constraints(predicate_node, self.maintain_errors, expr_arena)
        else {
            return Ok(None);
        };
        // The rewrite is always a fresh node (a `Literal(false)` or a rebuilt
        // chain). The check just stops the optimizer looping forever if that
        // ever changes.
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
