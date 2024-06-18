use polars_core::prelude::PolarsResult;

use crate::plans::aexpr::AExpr;
use crate::plans::ir::IR;
use crate::prelude::{Arena, Node};

/// Optimizer that uses a stack and memory arenas in favor of recursion
pub struct StackOptimizer {}

impl StackOptimizer {
    pub fn optimize_loop(
        &self,
        rules: &mut [Box<dyn OptimizationRule>],
        expr_arena: &mut Arena<AExpr>,
        lp_arena: &mut Arena<IR>,
        lp_top: Node,
    ) -> PolarsResult<Node> {
        let mut changed = true;

        // Nodes of expressions and lp node from which the expressions are a member of.
        let mut plans = vec![];
        let mut exprs = vec![];
        let mut scratch = vec![];

        // Run loop until reaching fixed point.
        while changed {
            // Recurse into sub plans and expressions and apply rules.
            changed = false;
            plans.push(lp_top);
            while let Some(current_node) = plans.pop() {
                // Apply rules
                for rule in rules.iter_mut() {
                    // keep iterating over same rule
                    while let Some(x) = rule.optimize_plan(lp_arena, expr_arena, current_node) {
                        lp_arena.replace(current_node, x);
                        changed = true;
                    }
                }

                let plan = lp_arena.get(current_node);

                // traverse subplans and expressions and add to the stack
                plan.copy_exprs(&mut scratch);
                plan.copy_inputs(&mut plans);

                if scratch.is_empty() {
                    continue;
                }

                while let Some(expr_ir) = scratch.pop() {
                    exprs.push(expr_ir.node());
                }

                // process the expressions on the stack and apply optimizations.
                while let Some(current_expr_node) = exprs.pop() {
                    {
                        let expr = unsafe { expr_arena.get_unchecked(current_expr_node) };
                        if expr.is_leaf() {
                            continue;
                        }
                    }
                    for rule in rules.iter_mut() {
                        // keep iterating over same rule
                        while let Some(x) = rule.optimize_expr(
                            expr_arena,
                            current_expr_node,
                            lp_arena,
                            current_node,
                        )? {
                            expr_arena.replace(current_expr_node, x);
                            changed = true;
                        }
                    }

                    let expr = unsafe { expr_arena.get_unchecked(current_expr_node) };
                    // traverse subexpressions and add to the stack
                    expr.nodes(&mut exprs)
                }
            }
        }
        Ok(lp_top)
    }
}

pub trait OptimizationRule {
    ///  Optimize (subplan) in LogicalPlan
    ///
    /// * `lp_arena` - LogicalPlan memory arena
    /// * `expr_arena` - Expression memory arena
    /// * `node` - node of the current LogicalPlan node
    fn optimize_plan(
        &mut self,
        _lp_arena: &mut Arena<IR>,
        _expr_arena: &mut Arena<AExpr>,
        _node: Node,
    ) -> Option<IR> {
        None
    }
    fn optimize_expr(
        &mut self,
        _expr_arena: &mut Arena<AExpr>,
        _expr_node: Node,
        _lp_arena: &Arena<IR>,
        _lp_node: Node,
    ) -> PolarsResult<Option<AExpr>> {
        Ok(None)
    }
}
