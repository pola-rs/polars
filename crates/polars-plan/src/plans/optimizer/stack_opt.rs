use std::sync::Arc;

use polars_core::prelude::PolarsResult;
use polars_core::schema::Schema;

use crate::constants::get_pl_structfields_name;
use crate::dsl::Operator;
use crate::plans::aexpr::AExpr;
use crate::plans::ir::IR;
use crate::plans::{ExprIR, ToFieldContext, get_input, get_input_schema};
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
        let mut children = vec![];
        let mut schema_stack: Vec<Arc<Schema>> = vec![];

        // Run loop until reaching fixed point.
        #[allow(clippy::field_reassign_with_default)]
        while changed {
            // Recurse into sub plans and expressions and apply rules.
            changed = false;

            plans.push(lp_top);
            while let Some(current_node) = plans.pop() {
                // Apply rules
                for rule in rules.iter_mut() {
                    // keep iterating over same rule
                    while let Some(x) = rule.optimize_plan(lp_arena, expr_arena, current_node)? {
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

                let input_schema = get_input_schema(lp_arena, current_node);

                schema_stack.clear();
                let schema_idx = 0;
                schema_stack.push(input_schema.into_owned());

                let is_filter = matches!(plan, IR::Filter { .. });
                while let Some(expr_ir) = scratch.pop() {
                    exprs.push((expr_ir.node(), schema_idx, is_filter));
                }

                let mut ctx = OptimizeExprContext::default();
                #[cfg(feature = "python")]
                {
                    use crate::dsl::python_dsl::PythonScanSource;
                    ctx.in_pyarrow_scan = matches!(plan, IR::PythonScan { options } if options.python_source == PythonScanSource::Pyarrow);
                    ctx.in_io_plugin = matches!(plan, IR::PythonScan { options } if options.python_source == PythonScanSource::IOPlugin);
                };
                ctx.has_inputs = !get_input(lp_arena, current_node).is_empty();

                // process the expressions on the stack and apply optimizations.
                while let Some((current_expr_node, schema_idx, is_filter_predicate)) = exprs.pop() {
                    ctx.is_filter_predicate = is_filter_predicate;
                    {
                        let expr = unsafe { expr_arena.get_unchecked(current_expr_node) };
                        if expr.is_leaf() {
                            continue;
                        }
                    }
                    let schema = &schema_stack[schema_idx];
                    for rule in rules.iter_mut() {
                        while let Some(x) =
                            rule.optimize_expr(expr_arena, current_expr_node, schema, ctx)?
                        {
                            expr_arena.replace(current_expr_node, x);
                            changed = true;
                        }
                    }
                    let expr = unsafe { expr_arena.get_unchecked(current_expr_node) };

                    // traverse subexpressions and add to the stack
                    match expr {
                        #[cfg(feature = "dtype-struct")]
                        AExpr::StructEval {
                            expr, evaluation, ..
                        } => {
                            let struct_field = expr_arena
                                .get(current_expr_node)
                                .to_field_impl(&ToFieldContext::new(expr_arena, schema))?;

                            let mut eval_schema = (**schema).clone();
                            eval_schema
                                .insert(get_pl_structfields_name(), struct_field.dtype().clone());
                            let eval_schema_idx = schema_stack.len();
                            schema_stack.push(Arc::new(eval_schema));

                            exprs.push((*expr, schema_idx, false));
                            for node in evaluation.iter().map(ExprIR::node).rev() {
                                exprs.push((node, eval_schema_idx, false));
                            }
                        },
                        expr => {
                            let is_filter_predicate =
                                is_filter_predicate && passes_on_filter_predicate(expr);
                            children.clear();
                            expr.inputs_rev(&mut children);
                            for node in children.drain(..) {
                                exprs.push((node, schema_idx, is_filter_predicate));
                            }
                        },
                    }
                }
            }
        }
        Ok(lp_top)
    }
}

#[derive(Default, Clone, Copy)]
pub struct OptimizeExprContext {
    pub in_pyarrow_scan: bool,
    pub in_io_plugin: bool,
    /// The expression is the predicate of a filter, or is reachable from it only through AND/OR.
    ///
    /// Such an expression only matters through whether it evaluates to `true`: a filter keeps a
    /// row iff its predicate is `true`, and under Kleene logic `a & b` is `true` iff both `a` and
    /// `b` are, and `a | b` iff either is. So rewrites may treat `null` and `false` as the same
    /// value here and may replace a column by a broadcasting scalar. Neither is valid for any
    /// other sub-expression of a filter, e.g. under `NOT`, `XOR`, `is_null` or an aggregation.
    pub is_filter_predicate: bool,
    pub has_inputs: bool,
}

/// Whether the inputs of `expr` are filter predicates if `expr` is one, see
/// [`OptimizeExprContext::is_filter_predicate`].
pub fn passes_on_filter_predicate(expr: &AExpr) -> bool {
    matches!(
        expr,
        AExpr::BinaryExpr {
            op: Operator::And | Operator::Or,
            ..
        }
    )
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
    ) -> PolarsResult<Option<IR>> {
        Ok(None)
    }
    fn optimize_expr(
        &mut self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        schema: &Schema,
        ctx: OptimizeExprContext,
    ) -> PolarsResult<Option<AExpr>> {
        _ = (expr_arena, expr_node, schema, ctx);
        Ok(None)
    }
}
