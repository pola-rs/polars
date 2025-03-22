use super::*;

pub(super) fn optimize_functions(
    input: &[ExprIR],
    function: &FunctionExpr,
    options: &FunctionOptions,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<Option<AExpr>> {
    let out = match function {
        // is_null().any() -> null_count() > 0
        // is_not_null().any() ->  null_count() < len()
        // CORRECTNESS: we can ignore 'ignore_nulls' since is_null/is_not_null never produces NULLS
        FunctionExpr::Boolean(BooleanFunction::Any { ignore_nulls: _ }) => {
            let input_node = expr_arena.get(input[0].node());
            match input_node {
                AExpr::Function {
                    input,
                    function: FunctionExpr::Boolean(BooleanFunction::IsNull),
                    options: _,
                } => Some(AExpr::BinaryExpr {
                    left: expr_arena.add(new_null_count(input)),
                    op: Operator::Gt,
                    right: expr_arena.add(AExpr::Literal(LiteralValue::new_idxsize(0))),
                }),
                AExpr::Function {
                    input,
                    function: FunctionExpr::Boolean(BooleanFunction::IsNotNull),
                    options: _,
                } => {
                    // we should perform optimization only if the original expression is a column
                    // so in case of disabled CSE, we will not suffer from performance regression
                    if input.len() == 1 {
                        let is_not_null_input_node = input[0].node();
                        match expr_arena.get(is_not_null_input_node) {
                            AExpr::Column(_) => Some(AExpr::BinaryExpr {
                                op: Operator::Lt,
                                left: expr_arena.add(new_null_count(input)),
                                right: expr_arena.add(AExpr::Agg(IRAggExpr::Count(
                                    is_not_null_input_node,
                                    true,
                                ))),
                            }),
                            _ => None,
                        }
                    } else {
                        None
                    }
                },
                _ => None,
            }
        },
        // is_null().all() -> null_count() == len()
        // is_not_null().all() -> null_count() == 0
        FunctionExpr::Boolean(BooleanFunction::All { ignore_nulls: _ }) => {
            let input_node = expr_arena.get(input[0].node());
            match input_node {
                AExpr::Function {
                    input,
                    function: FunctionExpr::Boolean(BooleanFunction::IsNull),
                    options: _,
                } => {
                    // we should perform optimization only if the original expression is a column
                    // so in case of disabled CSE, we will not suffer from performance regression
                    if input.len() == 1 {
                        let is_null_input_node = input[0].node();
                        match expr_arena.get(is_null_input_node) {
                            AExpr::Column(_) => Some(AExpr::BinaryExpr {
                                op: Operator::Eq,
                                right: expr_arena.add(new_null_count(input)),
                                left: expr_arena
                                    .add(AExpr::Agg(IRAggExpr::Count(is_null_input_node, true))),
                            }),
                            _ => None,
                        }
                    } else {
                        None
                    }
                },
                AExpr::Function {
                    input,
                    function: FunctionExpr::Boolean(BooleanFunction::IsNotNull),
                    options: _,
                } => Some(AExpr::BinaryExpr {
                    left: expr_arena.add(new_null_count(input)),
                    op: Operator::Eq,
                    right: expr_arena.add(AExpr::Literal(LiteralValue::new_idxsize(0))),
                }),
                _ => None,
            }
        },
        // sort().reverse() -> sort(reverse)
        // sort_by().reverse() -> sort_by(reverse)
        FunctionExpr::Reverse => {
            let input = expr_arena.get(input[0].node());
            match input {
                AExpr::Sort { expr, options } => {
                    let mut options = *options;
                    options.descending = !options.descending;
                    Some(AExpr::Sort {
                        expr: *expr,
                        options,
                    })
                },
                AExpr::SortBy {
                    expr,
                    by,
                    sort_options,
                } => {
                    let mut sort_options = sort_options.clone();
                    let reversed_descending = sort_options.descending.iter().map(|x| !*x).collect();
                    sort_options.descending = reversed_descending;
                    Some(AExpr::SortBy {
                        expr: *expr,
                        by: by.clone(),
                        sort_options,
                    })
                },
                // TODO: add support for cum_sum and other operation that allow reversing.
                _ => None,
            }
        },
        // flatten nested concat_str calls
        #[cfg(all(feature = "strings", feature = "concat_str"))]
        function @ FunctionExpr::StringExpr(StringFunction::ConcatHorizontal {
            delimiter: sep,
            ignore_nulls,
        }) if sep.is_empty() => {
            if input
                .iter()
                .any(|e| is_string_concat(expr_arena.get(e.node()), *ignore_nulls))
            {
                let mut new_inputs = Vec::with_capacity(input.len() * 2);

                for e in input {
                    match get_string_concat_input(e.node(), expr_arena, *ignore_nulls) {
                        Some(inp) => new_inputs.extend_from_slice(inp),
                        None => new_inputs.push(e.clone()),
                    }
                }
                Some(AExpr::Function {
                    input: new_inputs,
                    function: function.clone(),
                    options: *options,
                })
            } else {
                None
            }
        },
        FunctionExpr::Boolean(BooleanFunction::Not) => {
            let y = expr_arena.get(input[0].node());

            match y {
                // not(a and b) => not(a) or not(b)
                AExpr::BinaryExpr {
                    left,
                    op: Operator::And | Operator::LogicalAnd,
                    right,
                } => {
                    let left = *left;
                    let right = *right;
                    Some(AExpr::BinaryExpr {
                        left: expr_arena.add(AExpr::Function {
                            input: vec![ExprIR::from_node(left, expr_arena)],
                            function: FunctionExpr::Boolean(BooleanFunction::Not),
                            options: *options,
                        }),
                        op: Operator::Or,
                        right: expr_arena.add(AExpr::Function {
                            input: vec![ExprIR::from_node(right, expr_arena)],
                            function: FunctionExpr::Boolean(BooleanFunction::Not),
                            options: *options,
                        }),
                    })
                },
                // not(a or b) => not(a) and not(b)
                AExpr::BinaryExpr {
                    left,
                    op: Operator::Or | Operator::LogicalOr,
                    right,
                } => {
                    let left = *left;
                    let right = *right;
                    Some(AExpr::BinaryExpr {
                        left: expr_arena.add(AExpr::Function {
                            input: vec![ExprIR::from_node(left, expr_arena)],
                            function: FunctionExpr::Boolean(BooleanFunction::Not),
                            options: *options,
                        }),
                        op: Operator::And,
                        right: expr_arena.add(AExpr::Function {
                            input: vec![ExprIR::from_node(right, expr_arena)],
                            function: FunctionExpr::Boolean(BooleanFunction::Not),
                            options: *options,
                        }),
                    })
                },
                // not(not x) => x
                AExpr::Function {
                    input,
                    function: FunctionExpr::Boolean(BooleanFunction::Not),
                    ..
                } => Some(expr_arena.get(input[0].node()).clone()),
                // not(lit x) => !x
                AExpr::Literal(lv) if lv.bool().is_some() => {
                    Some(AExpr::Literal(Scalar::from(!lv.bool().unwrap()).into()))
                },
                // not(x.is_null) => x.is_not_null
                AExpr::Function {
                    input,
                    function: FunctionExpr::Boolean(BooleanFunction::IsNull),
                    options,
                } => Some(AExpr::Function {
                    input: input.clone(),
                    function: FunctionExpr::Boolean(BooleanFunction::IsNotNull),
                    options: *options,
                }),
                // not(x.is_not_null) => x.is_null
                AExpr::Function {
                    input,
                    function: FunctionExpr::Boolean(BooleanFunction::IsNotNull),
                    options,
                } => Some(AExpr::Function {
                    input: input.clone(),
                    function: FunctionExpr::Boolean(BooleanFunction::IsNull),
                    options: *options,
                }),
                // not(a == b) => a != b
                AExpr::BinaryExpr {
                    left,
                    op: Operator::Eq,
                    right,
                } => Some(AExpr::BinaryExpr {
                    left: *left,
                    op: Operator::NotEq,
                    right: *right,
                }),
                // not(a != b) => a == b
                AExpr::BinaryExpr {
                    left,
                    op: Operator::NotEq,
                    right,
                } => Some(AExpr::BinaryExpr {
                    left: *left,
                    op: Operator::Eq,
                    right: *right,
                }),
                // not(a < b) => a >= b
                AExpr::BinaryExpr {
                    left,
                    op: Operator::Lt,
                    right,
                } => Some(AExpr::BinaryExpr {
                    left: *left,
                    op: Operator::GtEq,
                    right: *right,
                }),
                // not(a <= b) => a > b
                AExpr::BinaryExpr {
                    left,
                    op: Operator::LtEq,
                    right,
                } => Some(AExpr::BinaryExpr {
                    left: *left,
                    op: Operator::Gt,
                    right: *right,
                }),
                // not(a > b) => a <= b
                AExpr::BinaryExpr {
                    left,
                    op: Operator::Gt,
                    right,
                } => Some(AExpr::BinaryExpr {
                    left: *left,
                    op: Operator::LtEq,
                    right: *right,
                }),
                // not(a >= b) => a < b
                AExpr::BinaryExpr {
                    left,
                    op: Operator::GtEq,
                    right,
                } => Some(AExpr::BinaryExpr {
                    left: *left,
                    op: Operator::Lt,
                    right: *right,
                }),
                #[cfg(feature = "is_between")]
                // not(col('x').is_between(a,b)) => col('x') < a || col('x') > b
                AExpr::Function {
                    input,
                    function: FunctionExpr::Boolean(BooleanFunction::IsBetween { closed }),
                    ..
                } => {
                    if !matches!(expr_arena.get(input[0].node()), AExpr::Column(_)) {
                        None
                    } else {
                        let left_cmp_op = match closed {
                            ClosedInterval::Both | ClosedInterval::Left => Operator::Lt,
                            ClosedInterval::None | ClosedInterval::Right => Operator::LtEq,
                        };
                        let right_cmp_op = match closed {
                            ClosedInterval::Both | ClosedInterval::Right => Operator::Gt,
                            ClosedInterval::None | ClosedInterval::Left => Operator::GtEq,
                        };
                        let left_left = input[0].node();
                        let right_left = input[1].node();

                        let left_right = left_left;
                        let right_right = input[2].node();

                        // input[0] is between input[1] and input[2]
                        Some(AExpr::BinaryExpr {
                            // input[0] (<,<=) input[1]
                            left: expr_arena.add(AExpr::BinaryExpr {
                                left: left_left,
                                op: left_cmp_op,
                                right: right_left,
                            }),
                            // OR
                            op: Operator::Or,
                            // input[0] (>,>=) input[2]
                            right: expr_arena.add(AExpr::BinaryExpr {
                                left: left_right,
                                op: right_cmp_op,
                                right: right_right,
                            }),
                        })
                    }
                },
                _ => None,
            }
        },
        _ => None,
    };
    Ok(out)
}

#[cfg(all(feature = "strings", feature = "concat_str"))]
fn is_string_concat(ae: &AExpr, ignore_nulls: bool) -> bool {
    matches!(ae, AExpr::Function {
                function:FunctionExpr::StringExpr(
                    StringFunction::ConcatHorizontal{delimiter: sep, ignore_nulls: func_inore_nulls},
                ),
                ..
            } if sep.is_empty() && *func_inore_nulls == ignore_nulls)
}

#[cfg(all(feature = "strings", feature = "concat_str"))]
fn get_string_concat_input(
    node: Node,
    expr_arena: &Arena<AExpr>,
    ignore_nulls: bool,
) -> Option<&[ExprIR]> {
    match expr_arena.get(node) {
        AExpr::Function {
            input,
            function:
                FunctionExpr::StringExpr(StringFunction::ConcatHorizontal {
                    delimiter: sep,
                    ignore_nulls: func_ignore_nulls,
                }),
            ..
        } if sep.is_empty() && *func_ignore_nulls == ignore_nulls => Some(input),
        _ => None,
    }
}
