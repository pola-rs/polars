use std::borrow::Borrow;

use self::type_check::TypeCheckRule;
use super::*;

/// Applies expression simplification and type coercion during conversion to IR.
pub struct ConversionOptimizer {
    scratch: Vec<(Node, usize)>,
    schemas: Vec<Schema>,

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

struct ExtendVec<'a> {
    out: &'a mut Vec<(Node, usize)>,
    schema_idx: usize,
}
impl Extend<Node> for ExtendVec<'_> {
    fn extend<T: IntoIterator<Item = Node>>(&mut self, iter: T) {
        self.out
            .extend(iter.into_iter().map(|n| (n, self.schema_idx)))
    }
}

impl ConversionOptimizer {
    pub fn new(simplify: bool, type_coercion: bool, type_check: bool) -> Self {
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
            schemas: Vec::new(),
            simplify,
            coerce,
            check,
            used_arenas: Default::default(),
        }
    }

    pub fn push_scratch(&mut self, expr: Node, expr_arena: &Arena<AExpr>) {
        self.scratch.push((expr, 0));
        // traverse all subexpressions and add to the stack
        let expr = unsafe { expr_arena.get_unchecked(expr) };
        expr.inputs_rev(&mut ExtendVec {
            out: &mut self.scratch,
            schema_idx: 0,
        });
    }

    pub fn fill_scratch<I, N>(&mut self, exprs: I, expr_arena: &Arena<AExpr>)
    where
        I: IntoIterator<Item = N>,
        N: Borrow<Node>,
    {
        for e in exprs {
            let node = *e.borrow();
            self.push_scratch(node, expr_arena);
        }
    }

    /// Optimizes the expressions in the scratch space. This should be called after filling the
    /// scratch space with the expressions that you want to optimize.
    pub fn optimize_exprs(
        &mut self,
        expr_arena: &mut Arena<AExpr>,
        ir_arena: &mut Arena<IR>,
        current_ir_node: Node,
        // Use the schema of `current_ir_node` instead of its input when resolving expr fields.
        use_current_node_schema: bool,
    ) -> PolarsResult<()> {
        // Different from the stack-opt in the optimizer phase, this does a single pass until fixed point per expression.

        if let Some(rule) = &mut self.check {
            while let Some(x) = rule.optimize_plan(ir_arena, expr_arena, current_ir_node)? {
                ir_arena.replace(current_ir_node, x);
            }
        }

        // process the expressions on the stack and apply optimizations.
        let schema = if use_current_node_schema {
            ir_arena.get(current_ir_node).schema(ir_arena)
        } else {
            get_input_schema(ir_arena, current_ir_node)
        };
        let plan = ir_arena.get(current_ir_node);
        let mut ctx = OptimizeExprContext {
            in_filter: matches!(plan, IR::Filter { .. }),
            has_inputs: !get_input(ir_arena, current_ir_node).is_empty(),
            ..Default::default()
        };
        #[cfg(feature = "python")]
        {
            use crate::dsl::python_dsl::PythonScanSource;
            ctx.in_pyarrow_scan = matches!(plan, IR::PythonScan { options } if options.python_source == PythonScanSource::Pyarrow);
            ctx.in_io_plugin = matches!(plan, IR::PythonScan { options } if options.python_source == PythonScanSource::IOPlugin);
        };

        self.schemas.clear();
        while let Some((current_expr_node, schema_idx)) = self.scratch.pop() {
            let expr = unsafe { expr_arena.get_unchecked(current_expr_node) };

            if expr.is_leaf() {
                continue;
            }

            // Evaluation expressions still need to do rules on the evaluation expression but the
            // schema is not the same and it is not concluded in the inputs. Therefore, we handl
            if let AExpr::Eval {
                expr,
                evaluation,
                variant,
            } = expr
            {
                let schema = if schema_idx == 0 {
                    &schema
                } else {
                    &self.schemas[schema_idx - 1]
                };
                let expr = expr_arena.get(*expr).get_dtype(schema, expr_arena)?;

                let element_dtype = variant.element_dtype(&expr)?;
                let schema = Schema::from_iter([(PlSmallStr::EMPTY, element_dtype.clone())]);
                self.schemas.push(schema);
                self.scratch.push((*evaluation, self.schemas.len()));
            }

            let schema = if schema_idx == 0 {
                &schema
            } else {
                &self.schemas[schema_idx - 1]
            };

            if let Some(rule) = &mut self.simplify {
                while let Some(x) =
                    rule.optimize_expr(expr_arena, current_expr_node, schema, ctx)?
                {
                    expr_arena.replace(current_expr_node, x);
                }
            }
            if let Some(rule) = &mut self.coerce {
                while let Some(x) =
                    rule.optimize_expr(expr_arena, current_expr_node, schema, ctx)?
                {
                    expr_arena.replace(current_expr_node, x);
                }
            }

            let expr = unsafe { expr_arena.get_unchecked(current_expr_node) };
            // traverse subexpressions and add to the stack
            expr.inputs_rev(&mut ExtendVec {
                out: &mut self.scratch,
                schema_idx,
            });
        }

        Ok(())
    }
}
