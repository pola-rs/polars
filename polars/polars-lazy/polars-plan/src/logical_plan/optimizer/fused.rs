use super::*;

pub struct FusedArithmetic {}

fn get_expr(input: Vec<Node>, op: FusedOperator) -> AExpr {
    AExpr::Function {
        input,
        function: FunctionExpr::Fused(op),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
            cast_to_supertypes: true,
            ..Default::default()
        },
    }
}

impl OptimizationRule for FusedArithmetic {
    #[allow(clippy::float_cmp)]
    fn optimize_expr(
        &self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        lp_arena: &Arena<ALogicalPlan>,
        lp_node: Node,
    ) -> PolarsResult<Option<AExpr>> {
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
                    } => {
                        let Some(input_node) = lp_arena.get(lp_node).get_input() else {return Ok(None)};
                        let schema = lp_arena.get(input_node).schema(lp_arena);
                        let type_left = expr_arena.get(*left).get_type(
                            &schema,
                            Context::Default,
                            expr_arena,
                        )?;
                        let type_right = expr_arena.get(*right).get_type(
                            &schema,
                            Context::Default,
                            expr_arena,
                        )?;
                        if type_left.is_numeric() && type_right.is_numeric() {
                            let input = vec![*right, *a, *b];
                            Ok(Some(get_expr(input, FusedOperator::MultiplyAdd)))
                        } else {
                            Ok(None)
                        }
                    }
                    _ => match expr_arena.get(*right) {
                        // input
                        // (a + (b * c)
                        // kept as input
                        BinaryExpr {
                            left: a,
                            op: Operator::Multiply,
                            right: b,
                        } => {
                            let Some(input_node) = lp_arena.get(lp_node).get_input() else {return Ok(None)};
                            let schema = lp_arena.get(input_node).schema(lp_arena);
                            let type_left = expr_arena.get(*left).get_type(
                                &schema,
                                Context::Default,
                                expr_arena,
                            )?;
                            let type_right = expr_arena.get(*right).get_type(
                                &schema,
                                Context::Default,
                                expr_arena,
                            )?;
                            if type_left.is_numeric() && type_right.is_numeric() {
                                let input = vec![*left, *a, *b];
                                Ok(Some(get_expr(input, FusedOperator::MultiplyAdd)))
                            } else {
                                Ok(None)
                            }
                        }
                        _ => Ok(None),
                    },
                }
            }

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
                    } => {
                        let Some(input_node) = lp_arena.get(lp_node).get_input() else {return Ok(None)};
                        let schema = lp_arena.get(input_node).schema(lp_arena);
                        let type_left = expr_arena.get(*left).get_type(
                            &schema,
                            Context::Default,
                            expr_arena,
                        )?;
                        let type_right = expr_arena.get(*right).get_type(
                            &schema,
                            Context::Default,
                            expr_arena,
                        )?;
                        if type_left.is_numeric() && type_right.is_numeric() {
                            let input = vec![*left, *a, *b];
                            Ok(Some(get_expr(input, FusedOperator::SubMultiply)))
                        } else {
                            Ok(None)
                        }
                    }
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
                            } => {
                                let Some(input_node) = lp_arena.get(lp_node).get_input() else {return Ok(None)};
                                let schema = lp_arena.get(input_node).schema(lp_arena);
                                let type_left = expr_arena.get(*left).get_type(
                                    &schema,
                                    Context::Default,
                                    expr_arena,
                                )?;
                                let type_right = expr_arena.get(*right).get_type(
                                    &schema,
                                    Context::Default,
                                    expr_arena,
                                )?;
                                if type_left.is_numeric() && type_right.is_numeric() {
                                    let input = vec![*a, *b, *right];
                                    Ok(Some(get_expr(input, FusedOperator::MultiplySub)))
                                } else {
                                    Ok(None)
                                }
                            }
                            _ => Ok(None),
                        }
                    }
                }
            }
            _ => Ok(None),
        }
    }
}
