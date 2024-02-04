use super::*;

pub(super) fn optimize_functions(
    input: &[Node],
    function: &FunctionExpr,
    _options: &FunctionOptions,
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<Option<AExpr>> {
    let out = match function {
        // sort().reverse() -> sort(reverse)
        // sort_by().reverse() -> sort_by(reverse)
        FunctionExpr::Reverse => {
            let input = expr_arena.get(input[0]);
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
                    descending,
                } => Some(AExpr::SortBy {
                    expr: *expr,
                    by: by.clone(),
                    descending: descending.iter().map(|r| !*r).collect(),
                }),
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
                .any(|node| is_string_concat(expr_arena.get(*node), *ignore_nulls))
            {
                let mut new_inputs = Vec::with_capacity(input.len() * 2);

                for node in input {
                    match get_string_concat_input(*node, expr_arena, *ignore_nulls) {
                        Some(inp) => new_inputs.extend_from_slice(inp),
                        None => new_inputs.push(*node),
                    }
                }
                Some(AExpr::Function {
                    input: new_inputs,
                    function: function.clone(),
                    options: *_options,
                })
            } else {
                None
            }
        },
        FunctionExpr::Boolean(BooleanFunction::AllHorizontal | BooleanFunction::AnyHorizontal) => {
            if input.len() == 1 {
                Some(AExpr::Cast {
                    expr: input[0],
                    data_type: DataType::Boolean,
                    strict: false,
                })
            } else {
                None
            }
        },
        FunctionExpr::Boolean(BooleanFunction::Not) => {
            let y = expr_arena.get(input[0]);

            match y {
                // not(not x) => x
                AExpr::Function {
                    input,
                    function: FunctionExpr::Boolean(BooleanFunction::Not),
                    ..
                } => Some(expr_arena.get(input[0]).clone()),
                // not(lit x) => !x
                AExpr::Literal(LiteralValue::Boolean(b)) => {
                    Some(AExpr::Literal(LiteralValue::Boolean(!b)))
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
) -> Option<&[Node]> {
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
