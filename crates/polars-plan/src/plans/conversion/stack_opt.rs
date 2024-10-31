use std::borrow::Borrow;

use super::*;

/// Applies expression simplification and type coercion during conversion to IR.
pub(super) struct ConversionOptimizer {
    scratch: Vec<Node>,
    simplify: Option<SimplifyExprRule>,
    coerce: Option<TypeCoercionRule>,
    // IR's can be cached in the DSL.
    // But if they are used multiple times in DSL (e.g. concat/join)
    // then it can occur that we take a slot multiple times.
    // So we keep track of the arena versions used and allow only
    // one unique IR cache to be reused.
    pub(super) used_arenas: PlHashSet<u32>,
}

impl ConversionOptimizer {
    pub(super) fn new(simplify: bool, type_coercion: bool) -> Self {
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

        ConversionOptimizer {
            scratch: Vec::with_capacity(8),
            simplify,
            coerce,
            used_arenas: Default::default(),
        }
    }

    pub(super) fn push_scratch(&mut self, expr: Node, expr_arena: &Arena<AExpr>) {
        self.scratch.push(expr);
        // traverse all subexpressions and add to the stack
        let expr = unsafe { expr_arena.get_unchecked(expr) };
        expr.nodes(&mut self.scratch);
    }

    pub(super) fn fill_scratch<N: Borrow<Node>>(&mut self, exprs: &[N], expr_arena: &Arena<AExpr>) {
        for e in exprs {
            let node = *e.borrow();
            self.push_scratch(node, expr_arena);
        }
    }

    pub(super) fn coerce_types(
        &mut self,
        expr_arena: &mut Arena<AExpr>,
        lp_arena: &Arena<IR>,
        current_node: Node,
    ) -> PolarsResult<()> {
        // Different from the stack-opt in the optimizer phase, this does a single pass until fixed point per expression.

        // process the expressions on the stack and apply optimizations.
        while let Some(current_expr_node) = self.scratch.pop() {
            {
                let expr = unsafe { expr_arena.get_unchecked(current_expr_node) };
                if expr.is_leaf() {
                    continue;
                }
            }
            if let Some(rule) = &mut self.simplify {
                while let Some(x) =
                    rule.optimize_expr(expr_arena, current_expr_node, lp_arena, current_node)?
                {
                    expr_arena.replace(current_expr_node, x);
                }
            }
            if let Some(rule) = &mut self.coerce {
                while let Some(x) =
                    rule.optimize_expr(expr_arena, current_expr_node, lp_arena, current_node)?
                {
                    expr_arena.replace(current_expr_node, x);
                }
            }

            let expr = unsafe { expr_arena.get_unchecked(current_expr_node) };
            // traverse subexpressions and add to the stack
            expr.nodes(&mut self.scratch)
        }

        Ok(())
    }
}
