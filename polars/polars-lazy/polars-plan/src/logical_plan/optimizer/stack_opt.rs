use polars_core::prelude::PolarsResult;

use crate::logical_plan::aexpr::AExpr;
use crate::logical_plan::alp::ALogicalPlan;
use crate::prelude::{Arena, Node};

/// Optimizer that uses a stack and memory arenas in favor of recursion
pub struct StackOptimizer {}

impl StackOptimizer {
    pub fn optimize_loop(
        &self,
        rules: &mut [Box<dyn OptimizationRule>],
        expr_arena: &mut Arena<AExpr>,
        lp_arena: &mut Arena<ALogicalPlan>,
        lp_top: Node,
    ) -> PolarsResult<Node> {
        let mut changed = true;

        let mut plans = Vec::with_capacity(32);

        // nodes of expressions and lp node from which the expressions are a member of
        let mut exprs = Vec::with_capacity(32);
        let mut scratch = vec![];

        // run loop until reaching fixed point
        while changed {
            // recurse into sub plans and expressions and apply rules
            changed = false;
            plans.push(lp_top);
            while let Some(current_node) = plans.pop() {
                // apply rules
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

                // first do a single pass to ensure we process
                // from leaves to root.
                // this ensures for instance
                // that we first do constant folding on operands
                // before we decide that multiple binary expression
                // can be replaced with a fused operator
                while let Some(expr_node) = scratch.pop() {
                    exprs.push(expr_node);
                    // traverse all subexpressions and add to the stack
                    let expr = unsafe { expr_arena.get_unchecked(expr_node) };
                    expr.nodes(&mut exprs);
                }

                // process the expressions on the stack and apply optimizations.
                while let Some(current_expr_node) = exprs.pop() {
                    {
                        let expr = unsafe { expr_arena.get_unchecked(current_expr_node) };
                        // don't apply rules to `col`, `lit` etc.
                        if expr.is_leaf() {
                            continue;
                        }
                    }
                    for rule in rules.iter() {
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
        _lp_arena: &mut Arena<ALogicalPlan>,
        _expr_arena: &mut Arena<AExpr>,
        _node: Node,
    ) -> Option<ALogicalPlan> {
        None
    }
    fn optimize_expr(
        &self,
        _expr_arena: &mut Arena<AExpr>,
        _expr_node: Node,
        _lp_arena: &Arena<ALogicalPlan>,
        _lp_node: Node,
    ) -> PolarsResult<Option<AExpr>> {
        Ok(None)
    }
}
