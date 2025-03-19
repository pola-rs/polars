use std::borrow::Borrow;

use self::type_check::TypeCheckRule;
use super::*;

/// Applies expression simplification and type coercion during conversion to IR.
pub(super) struct ConversionOptimizer {
    scratch: Vec<Node>,

    simplify: Option<SimplifyExprRule>,
    coerce: Option<TypeCoercionRule>,
    check: Option<TypeCheckRule>,
    // IR's can be cached in the DSL.
    // But if they are used multiple times in DSL (e.g. concat/join)
    // then it can occur that we take a slot multiple times.
    // So we keep track of the arena versions used and allow only
    // one unique IR cache to be reused.
    pub(super) used_arenas: PlHashSet<u32>,
}

impl ConversionOptimizer {
    pub(super) fn new(simplify: bool, type_coercion: bool, type_check: bool) -> Self {
        let simplify = if simplify {
            Some(SimplifyExprRule {})
        } else {
            None
        };

        let coerce = if type_coercion {
            Some(TypeCoercionRule {})
        } else {
            None
        };

        let check = if type_check {
            Some(TypeCheckRule)
        } else {
            None
        };

        ConversionOptimizer {
            scratch: Vec::with_capacity(8),
            simplify,
            coerce,
            check,
            used_arenas: Default::default(),
        }
    }

    pub(super) fn push_scratch(&mut self, expr: Node, expr_arena: &Arena<AExpr>) {
        self.scratch.push(expr);
        // traverse all subexpressions and add to the stack
        let expr = unsafe { expr_arena.get_unchecked(expr) };
        expr.inputs_rev(&mut self.scratch);
    }

    pub(super) fn fill_scratch<N: Borrow<Node>>(&mut self, exprs: &[N], expr_arena: &Arena<AExpr>) {
        for e in exprs {
            let node = *e.borrow();
            self.push_scratch(node, expr_arena);
        }
    }

    /// Optimizes the expressions in the scratch space. This should be called after filling the
    /// scratch space with the expressions that you want to optimize.
    pub(super) fn optimize_exprs(
        &mut self,
        expr_arena: &mut Arena<AExpr>,
        ir_arena: &mut Arena<IR>,
        current_ir_node: Node,
    ) -> PolarsResult<()> {
        // Different from the stack-opt in the optimizer phase, this does a single pass until fixed point per expression.

        if let Some(rule) = &mut self.check {
            while let Some(x) = rule.optimize_plan(ir_arena, expr_arena, current_ir_node)? {
                ir_arena.replace(current_ir_node, x);
            }
        }

        // process the expressions on the stack and apply optimizations.
        while let Some(current_expr_node) = self.scratch.pop() {
            let expr = unsafe { expr_arena.get_unchecked(current_expr_node) };

            if expr.is_leaf() {
                continue;
            }

            if let Some(rule) = &mut self.simplify {
                while let Some(x) =
                    rule.optimize_expr(expr_arena, current_expr_node, ir_arena, current_ir_node)?
                {
                    expr_arena.replace(current_expr_node, x);
                }
            }
            if let Some(rule) = &mut self.coerce {
                while let Some(x) =
                    rule.optimize_expr(expr_arena, current_expr_node, ir_arena, current_ir_node)?
                {
                    expr_arena.replace(current_expr_node, x);
                }
            }

            let expr = unsafe { expr_arena.get_unchecked(current_expr_node) };
            // traverse subexpressions and add to the stack
            expr.inputs_rev(&mut self.scratch)
        }

        Ok(())
    }
}
