use super::stack_opt::OptimizeExprContext;
use super::*;

pub struct FusedArithmetic {}

fn get_expr(input: &[Node], op: FusedOperator, expr_arena: &Arena<AExpr>) -> AExpr {
    let input = input
        .iter()
        .copied()
        .map(|n| ExprIR::from_node(n, expr_arena))
        .collect();
    let mut options =
        FunctionOptions::elementwise().with_casting_rules(CastingRules::cast_to_supertypes());
    // order of operations change because of FMA
    // so we must toggle this check off
    // it is still safe as it is a trusted operation
    unsafe { options.no_check_lengths() }
    AExpr::Function {
        input,
        function: IRFunctionExpr::Fused(op),
        options,
    }
}

fn check_eligible(
    left: &Node,
    right: &Node,
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<bool> {
    let field_left = expr_arena.get(*left).to_field(schema, expr_arena)?;
    let type_right = expr_arena.get(*right).get_dtype(schema, expr_arena)?;
    let type_left = &field_left.dtype;
    // Exclude literals for now as these will not benefit from fused operations downstream #9857
    // This optimization would also interfere with the `col -> lit` type-coercion rules
    // And it might also interfere with constant folding which is a more suitable optimizations here
    if type_left.is_primitive_numeric()
        && type_right.is_primitive_numeric()
        && !has_aexpr_literal(*left, expr_arena)
        && !has_aexpr_literal(*right, expr_arena)
    {
        Ok(true)
    } else {
        Ok(false)
    }
}

impl OptimizationRule for FusedArithmetic {
    #[allow(clippy::float_cmp)]
    fn optimize_expr(
        &mut self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        schema: &Schema,
        ctx: OptimizeExprContext,
    ) -> PolarsResult<Option<AExpr>> {
        // We don't want to fuse arithmetic that we send to pyarrow.
        if ctx.in_pyarrow_scan || ctx.in_io_plugin {
            return Ok(None);
        }

        let expr = expr_arena.get(expr_node);

        use AExpr::*;
        match expr {
            BinaryExpr {
                left,
                op: Operator::Plus,
                right,
            } => {
                // FUSED MULTIPLY ADD
                // For fma the plus is always the out as the multiply takes prevalence
                match expr_arena.get(*left) {
                    // Argument order is a + b * c
                    // so we must swap operands
                    //
                    // input
                    // (a * b) + c
                    // swapped as
                    // c + (a * b)
                    BinaryExpr {
                        left: a,
                        op: Operator::Multiply,
                        right: b,
                    } => Ok(check_eligible(left, right, expr_arena, schema)?.then(|| {
                        let input = &[*right, *a, *b];
                        get_expr(input, FusedOperator::MultiplyAdd, expr_arena)
                    })),
                    _ => match expr_arena.get(*right) {
                        // input
                        // (a + (b * c)
                        // kept as input
                        BinaryExpr {
                            left: a,
                            op: Operator::Multiply,
                            right: b,
                        } => Ok(check_eligible(left, right, expr_arena, schema)?.then(|| {
                            let input = &[*left, *a, *b];
                            get_expr(input, FusedOperator::MultiplyAdd, expr_arena)
                        })),
                        _ => Ok(None),
                    },
                }
            },

            BinaryExpr {
                left,
                op: Operator::Minus,
                right,
            } => {
                // FUSED SUB MULTIPLY
                match expr_arena.get(*right) {
                    // input
                    // (a - (b * c)
                    // kept as input
                    BinaryExpr {
                        left: a,
                        op: Operator::Multiply,
                        right: b,
                    } => Ok(check_eligible(left, right, expr_arena, schema)?.then(|| {
                        let input = &[*left, *a, *b];
                        get_expr(input, FusedOperator::SubMultiply, expr_arena)
                    })),
                    _ => {
                        // FUSED MULTIPLY SUB
                        match expr_arena.get(*left) {
                            // input
                            // (a * b) - c
                            // kept as input
                            BinaryExpr {
                                left: a,
                                op: Operator::Multiply,
                                right: b,
                            } => Ok(check_eligible(left, right, expr_arena, schema)?.then(|| {
                                let input = &[*a, *b, *right];
                                get_expr(input, FusedOperator::MultiplySub, expr_arena)
                            })),
                            _ => Ok(None),
                        }
                    },
                }
            },
            _ => Ok(None),
        }
    }
}
