use super::*;

pub struct FusedArithmetic {}

fn get_expr(input: &[Node], op: FusedOperator, expr_arena: &Arena<AExpr>) -> AExpr {
    let input = input
        .iter()
        .copied()
        .map(|n| ExprIR::from_node(n, expr_arena))
        .collect();
    let mut options = FunctionOptions {
        collect_groups: ApplyOptions::ElementWise,
        cast_to_supertypes: Some(Default::default()),
        ..Default::default()
    };
    // order of operations change because of FMA
    // so we must toggle this check off
    // it is still safe as it is a trusted operation
    unsafe { options.no_check_lengths() }
    AExpr::Function {
        input,
        function: FunctionExpr::Fused(op),
        options,
    }
}

fn check_eligible(
    left: &Node,
    right: &Node,
    lp_node: Node,
    expr_arena: &Arena<AExpr>,
    lp_arena: &Arena<IR>,
) -> PolarsResult<(Option<bool>, Option<Field>)> {
    let Some(input_node) = lp_arena.get(lp_node).get_input() else {
        return Ok((None, None));
    };
    let schema = lp_arena.get(input_node).schema(lp_arena);
    let field_left = expr_arena
        .get(*left)
        .to_field(&schema, Context::Default, expr_arena)?;
    let type_right = expr_arena
        .get(*right)
        .get_type(&schema, Context::Default, expr_arena)?;
    let type_left = &field_left.dtype;
    // Exclude literals for now as these will not benefit from fused operations downstream #9857
    // This optimization would also interfere with the `col -> lit` type-coercion rules
    // And it might also interfere with constant folding which is a more suitable optimizations here
    if type_left.is_numeric()
        && type_right.is_numeric()
        && !has_aexpr_literal(*left, expr_arena)
        && !has_aexpr_literal(*right, expr_arena)
    {
        Ok((Some(true), Some(field_left)))
    } else {
        Ok((Some(false), None))
    }
}

impl OptimizationRule for FusedArithmetic {
    #[allow(clippy::float_cmp)]
    fn optimize_expr(
        &mut self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        lp_arena: &Arena<IR>,
        lp_node: Node,
    ) -> PolarsResult<Option<AExpr>> {
        // We don't want to fuse arithmetic that we send to pyarrow.
        #[cfg(feature = "python")]
        if let IR::PythonScan { options } = lp_arena.get(lp_node) {
            if matches!(
                options.python_source,
                PythonScanSource::Pyarrow | PythonScanSource::IOPlugin
            ) {
                return Ok(None);
            }
        };
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
                    } => match check_eligible(left, right, lp_node, expr_arena, lp_arena)? {
                        (None, _) | (Some(false), _) => Ok(None),
                        (Some(true), Some(output_field)) => {
                            let input = &[*right, *a, *b];
                            let fma = get_expr(input, FusedOperator::MultiplyAdd, expr_arena);
                            let node = expr_arena.add(fma);
                            // we reordered the arguments, so we don't obey the left expression output name
                            // rule anymore, that's why we alias
                            Ok(Some(Alias(
                                node,
                                ColumnName::from(output_field.name.as_str()),
                            )))
                        },
                        _ => unreachable!(),
                    },
                    _ => match expr_arena.get(*right) {
                        // input
                        // (a + (b * c)
                        // kept as input
                        BinaryExpr {
                            left: a,
                            op: Operator::Multiply,
                            right: b,
                        } => match check_eligible(left, right, lp_node, expr_arena, lp_arena)? {
                            (None, _) | (Some(false), _) => Ok(None),
                            (Some(true), _) => {
                                let input = &[*left, *a, *b];
                                Ok(Some(get_expr(
                                    input,
                                    FusedOperator::MultiplyAdd,
                                    expr_arena,
                                )))
                            },
                        },
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
                    } => match check_eligible(left, right, lp_node, expr_arena, lp_arena)? {
                        (None, _) | (Some(false), _) => Ok(None),
                        (Some(true), _) => {
                            let input = &[*left, *a, *b];
                            Ok(Some(get_expr(
                                input,
                                FusedOperator::SubMultiply,
                                expr_arena,
                            )))
                        },
                    },
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
                                match check_eligible(left, right, lp_node, expr_arena, lp_arena)? {
                                    (None, _) | (Some(false), _) => Ok(None),
                                    (Some(true), _) => {
                                        let input = &[*a, *b, *right];
                                        Ok(Some(get_expr(
                                            input,
                                            FusedOperator::MultiplySub,
                                            expr_arena,
                                        )))
                                    },
                                }
                            },
                            _ => Ok(None),
                        }
                    },
                }
            },
            _ => Ok(None),
        }
    }
}
