use polars_utils::arena::Arena;

#[cfg(all(feature = "strings", feature = "concat_str"))]
use crate::dsl::function_expr::StringFunction;
use crate::logical_plan::optimizer::stack_opt::OptimizationRule;
use crate::logical_plan::*;
use crate::prelude::function_expr::FunctionExpr;

macro_rules! eval_binary_same_type {
    ($lhs:expr, $operand: tt, $rhs:expr) => {{
    if let (AExpr::Literal(lit_left), AExpr::Literal(lit_right)) = ($lhs, $rhs) {
        match (lit_left, lit_right) {
            (LiteralValue::Float32(x), LiteralValue::Float32(y)) => {
                Some(AExpr::Literal(LiteralValue::Float32(x $operand y)))
            }
            (LiteralValue::Float64(x), LiteralValue::Float64(y)) => {
                Some(AExpr::Literal(LiteralValue::Float64(x $operand y)))
            }
            #[cfg(feature = "dtype-i8")]
            (LiteralValue::Int8(x), LiteralValue::Int8(y)) => {
                Some(AExpr::Literal(LiteralValue::Int8(x $operand y)))
            }
            #[cfg(feature = "dtype-i16")]
            (LiteralValue::Int16(x), LiteralValue::Int16(y)) => {
                Some(AExpr::Literal(LiteralValue::Int16(x $operand y)))
            }
            (LiteralValue::Int32(x), LiteralValue::Int32(y)) => {
                Some(AExpr::Literal(LiteralValue::Int32(x $operand y)))
            }
            (LiteralValue::Int64(x), LiteralValue::Int64(y)) => {
                Some(AExpr::Literal(LiteralValue::Int64(x $operand y)))
            }
            #[cfg(feature = "dtype-u8")]
            (LiteralValue::UInt8(x), LiteralValue::UInt8(y)) => {
                Some(AExpr::Literal(LiteralValue::UInt8(x $operand y)))
            }
            #[cfg(feature = "dtype-u16")]
            (LiteralValue::UInt16(x), LiteralValue::UInt16(y)) => {
                Some(AExpr::Literal(LiteralValue::UInt16(x $operand y)))
            }
            (LiteralValue::UInt32(x), LiteralValue::UInt32(y)) => {
                Some(AExpr::Literal(LiteralValue::UInt32(x $operand y)))
            }
            (LiteralValue::UInt64(x), LiteralValue::UInt64(y)) => {
                Some(AExpr::Literal(LiteralValue::UInt64(x $operand y)))
            }
            _ => None,
        }
    } else {
      None
    }

    }}
}

macro_rules! eval_binary_bool_type {
    ($lhs:expr, $operand: tt, $rhs:expr) => {{
    if let (AExpr::Literal(lit_left), AExpr::Literal(lit_right)) = ($lhs, $rhs) {
        match (lit_left, lit_right) {
            (LiteralValue::Float32(x), LiteralValue::Float32(y)) => {
                Some(AExpr::Literal(LiteralValue::Boolean(x $operand y)))
            }
            (LiteralValue::Float64(x), LiteralValue::Float64(y)) => {
                Some(AExpr::Literal(LiteralValue::Boolean(x $operand y)))
            }
            #[cfg(feature = "dtype-i8")]
            (LiteralValue::Int8(x), LiteralValue::Int8(y)) => {
                Some(AExpr::Literal(LiteralValue::Boolean(x $operand y)))
            }
            #[cfg(feature = "dtype-i16")]
            (LiteralValue::Int16(x), LiteralValue::Int16(y)) => {
                Some(AExpr::Literal(LiteralValue::Boolean(x $operand y)))
            }
            (LiteralValue::Int32(x), LiteralValue::Int32(y)) => {
                Some(AExpr::Literal(LiteralValue::Boolean(x $operand y)))
            }
            (LiteralValue::Int64(x), LiteralValue::Int64(y)) => {
                Some(AExpr::Literal(LiteralValue::Boolean(x $operand y)))
            }
            #[cfg(feature = "dtype-u8")]
            (LiteralValue::UInt8(x), LiteralValue::UInt8(y)) => {
                Some(AExpr::Literal(LiteralValue::Boolean(x $operand y)))
            }
            #[cfg(feature = "dtype-u16")]
            (LiteralValue::UInt16(x), LiteralValue::UInt16(y)) => {
                Some(AExpr::Literal(LiteralValue::Boolean(x $operand y)))
            }
            (LiteralValue::UInt32(x), LiteralValue::UInt32(y)) => {
                Some(AExpr::Literal(LiteralValue::Boolean(x $operand y)))
            }
            (LiteralValue::UInt64(x), LiteralValue::UInt64(y)) => {
                Some(AExpr::Literal(LiteralValue::Boolean(x $operand y)))
            }
            (LiteralValue::Boolean(x), LiteralValue::Boolean(y)) => {
                Some(AExpr::Literal(LiteralValue::Boolean(x $operand y)))
            }
            _ => None,
        }
    } else {
        None
    }

    }}
}

pub struct SimplifyBooleanRule {}

impl OptimizationRule for SimplifyBooleanRule {
    fn optimize_expr(
        &mut self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        _: &Arena<ALogicalPlan>,
        _: Node,
    ) -> PolarsResult<Option<AExpr>> {
        let expr = expr_arena.get(expr_node);
        let out = match expr {
            // true AND x => x
            AExpr::BinaryExpr {
                left,
                op: Operator::And,
                right,
            } if matches!(
                expr_arena.get(*left),
                AExpr::Literal(LiteralValue::Boolean(true))
            ) =>
            {
                // We alias because of the left-hand naming rule.
                Some(AExpr::Alias(*right, "literal".into()))
            },
            // x AND true => x
            AExpr::BinaryExpr {
                left,
                op: Operator::And,
                right,
            } if matches!(
                expr_arena.get(*right),
                AExpr::Literal(LiteralValue::Boolean(true))
            ) =>
            {
                Some(expr_arena.get(*left).clone())
            },

            // x AND false -> false
            // FIXME: we need an optimizer redesign to allow x & false to be optimized
            // in general as we can forget the length of a series otherwise.
            AExpr::BinaryExpr {
                left,
                op: Operator::And,
                right,
            } if matches!(expr_arena.get(*left), AExpr::Literal(_))
                && matches!(
                    expr_arena.get(*right),
                    AExpr::Literal(LiteralValue::Boolean(false))
                ) =>
            {
                Some(AExpr::Literal(LiteralValue::Boolean(false)))
            },

            // false AND x -> false
            // FIXME: we need an optimizer redesign to allow false & x to be optimized
            // in general as we can forget the length of a series otherwise.
            AExpr::BinaryExpr {
                left,
                op: Operator::And,
                right,
            } if matches!(
                expr_arena.get(*left),
                AExpr::Literal(LiteralValue::Boolean(false))
            ) && matches!(expr_arena.get(*right), AExpr::Literal(_)) =>
            {
                Some(AExpr::Literal(LiteralValue::Boolean(false)))
            },

            // false or x => x
            AExpr::BinaryExpr {
                left,
                op: Operator::Or,
                right,
            } if matches!(
                expr_arena.get(*left),
                AExpr::Literal(LiteralValue::Boolean(false))
            ) =>
            {
                // We alias because of the left-hand naming rule.
                Some(AExpr::Alias(*right, "literal".into()))
            },
            // x or false => x
            AExpr::BinaryExpr {
                left,
                op: Operator::Or,
                right,
                ..
            } if matches!(
                expr_arena.get(*right),
                AExpr::Literal(LiteralValue::Boolean(false))
            ) =>
            {
                Some(expr_arena.get(*left).clone())
            },

            // true OR x => true
            // FIXME: we need an optimizer redesign to allow true | x to be optimized
            // in general as we can forget the length of a series otherwise.
            AExpr::BinaryExpr {
                left,
                op: Operator::Or,
                right,
            } if matches!(expr_arena.get(*left), AExpr::Literal(_))
                && matches!(
                    expr_arena.get(*right),
                    AExpr::Literal(LiteralValue::Boolean(true))
                ) =>
            {
                Some(AExpr::Literal(LiteralValue::Boolean(true)))
            },

            // x OR true => true
            // FIXME: we need an optimizer redesign to allow true | x to be optimized
            // in general as we can forget the length of a series otherwise.
            AExpr::BinaryExpr {
                left,
                op: Operator::Or,
                right,
            } if matches!(
                expr_arena.get(*left),
                AExpr::Literal(LiteralValue::Boolean(true))
            ) && matches!(expr_arena.get(*right), AExpr::Literal(_)) =>
            {
                Some(AExpr::Literal(LiteralValue::Boolean(true)))
            },

            AExpr::Ternary {
                truthy, predicate, ..
            } if matches!(
                expr_arena.get(*predicate),
                AExpr::Literal(LiteralValue::Boolean(true))
            ) =>
            {
                Some(expr_arena.get(*truthy).clone())
            },
            AExpr::Ternary {
                truthy,
                falsy,
                predicate,
            } if matches!(
                expr_arena.get(*predicate),
                AExpr::Literal(LiteralValue::Boolean(false))
            ) =>
            {
                let names = aexpr_to_leaf_names(*truthy, expr_arena);
                let name = names.get(0).map(Arc::clone).unwrap_or_else(|| "".into());
                Some(AExpr::Alias(*falsy, name))
            },
            AExpr::Function {
                input,
                function: FunctionExpr::Boolean(BooleanFunction::Not),
                ..
            } => {
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
}

fn eval_bitwise<F>(left: &AExpr, right: &AExpr, operation: F) -> Option<AExpr>
where
    F: Fn(bool, bool) -> bool,
{
    if let (AExpr::Literal(lit_left), AExpr::Literal(lit_right)) = (left, right) {
        return match (lit_left, lit_right) {
            (LiteralValue::Boolean(x), LiteralValue::Boolean(y)) => {
                Some(AExpr::Literal(LiteralValue::Boolean(operation(*x, *y))))
            },
            _ => None,
        };
    }
    None
}

#[cfg(all(feature = "strings", feature = "concat_str"))]
fn string_addition_to_linear_concat(
    lp_arena: &Arena<ALogicalPlan>,
    lp_node: Node,
    expr_arena: &Arena<AExpr>,
    left_ae: Node,
    right_ae: Node,
    left_aexpr: &AExpr,
    right_aexpr: &AExpr,
) -> Option<AExpr> {
    {
        let lp = lp_arena.get(lp_node);
        let input = lp.get_input()?;
        let schema = lp_arena.get(input).schema(lp_arena);

        let get_type = |ae: &AExpr| ae.get_type(&schema, Context::Default, expr_arena).ok();
        let type_a = get_type(left_aexpr)
            .or_else(|| get_type(right_aexpr))
            .unwrap();
        let type_b = get_type(right_aexpr)
            .or_else(|| get_type(right_aexpr))
            .unwrap();

        if type_a != type_b {
            return None;
        }

        if type_a == DataType::Utf8 {
            match (left_aexpr, right_aexpr) {
                // concat + concat
                (
                    AExpr::Function {
                        input: input_left,
                        function:
                            ref
                            fun_l @ FunctionExpr::StringExpr(StringFunction::ConcatHorizontal(sep_l)),
                        options,
                    },
                    AExpr::Function {
                        input: input_right,
                        function: FunctionExpr::StringExpr(StringFunction::ConcatHorizontal(sep_r)),
                        ..
                    },
                ) => {
                    if sep_l.is_empty() && sep_r.is_empty() {
                        let mut input = Vec::with_capacity(input_left.len() + input_right.len());
                        input.extend_from_slice(input_left);
                        input.extend_from_slice(input_right);
                        Some(AExpr::Function {
                            input,
                            function: fun_l.clone(),
                            options: *options,
                        })
                    } else {
                        None
                    }
                },
                // concat + str
                (
                    AExpr::Function {
                        input,
                        function:
                            ref fun @ FunctionExpr::StringExpr(StringFunction::ConcatHorizontal(sep)),
                        options,
                    },
                    _,
                ) => {
                    if sep.is_empty() {
                        let mut input = input.clone();
                        input.push(right_ae);
                        Some(AExpr::Function {
                            input,
                            function: fun.clone(),
                            options: *options,
                        })
                    } else {
                        None
                    }
                },
                // str + concat
                (
                    _,
                    AExpr::Function {
                        input: input_right,
                        function:
                            ref fun @ FunctionExpr::StringExpr(StringFunction::ConcatHorizontal(sep)),
                        options,
                    },
                ) => {
                    if sep.is_empty() {
                        let mut input = Vec::with_capacity(1 + input_right.len());
                        input.push(left_ae);
                        input.extend_from_slice(input_right);
                        Some(AExpr::Function {
                            input,
                            function: fun.clone(),
                            options: *options,
                        })
                    } else {
                        None
                    }
                },
                _ => Some(AExpr::Function {
                    input: vec![left_ae, right_ae],
                    function: StringFunction::ConcatHorizontal("".to_string()).into(),
                    options: FunctionOptions {
                        collect_groups: ApplyOptions::ElementWise,
                        input_wildcard_expansion: true,
                        returns_scalar: false,
                        ..Default::default()
                    },
                }),
            }
        } else {
            None
        }
    }
}

#[cfg(all(feature = "strings", feature = "concat_str"))]
fn is_string_concat(ae: &AExpr) -> bool {
    matches!(ae, AExpr::Function {
                function:FunctionExpr::StringExpr(
                    StringFunction::ConcatHorizontal(sep),
                ),
                ..
            } if sep.is_empty())
}

#[cfg(all(feature = "strings", feature = "concat_str"))]
fn get_string_concat_input(node: Node, expr_arena: &Arena<AExpr>) -> Option<&[Node]> {
    match expr_arena.get(node) {
        AExpr::Function {
            input,
            function: FunctionExpr::StringExpr(StringFunction::ConcatHorizontal(sep)),
            ..
        } if sep.is_empty() => Some(input),
        _ => None,
    }
}

pub struct SimplifyExprRule {}

impl OptimizationRule for SimplifyExprRule {
    #[allow(clippy::float_cmp)]
    fn optimize_expr(
        &mut self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        _lp_arena: &Arena<ALogicalPlan>,
        _lp_node: Node,
    ) -> PolarsResult<Option<AExpr>> {
        let expr = expr_arena.get(expr_node);

        let out = match expr {
            // lit(left) + lit(right) => lit(left + right)
            // and null propagation
            AExpr::BinaryExpr { left, op, right } => {
                let left_aexpr = expr_arena.get(*left);
                let right_aexpr = expr_arena.get(*right);

                // lit(left) + lit(right) => lit(left + right)
                #[allow(clippy::manual_map)]
                let out = match op {
                    Plus => {
                        match eval_binary_same_type!(left_aexpr, +, right_aexpr) {
                            Some(new) => Some(new),
                            None => {
                                // try to replace addition of string columns with `concat_str`
                                #[cfg(all(feature = "strings", feature = "concat_str"))]
                                {
                                    string_addition_to_linear_concat(
                                        _lp_arena,
                                        _lp_node,
                                        expr_arena,
                                        *left,
                                        *right,
                                        left_aexpr,
                                        right_aexpr,
                                    )
                                }
                                #[cfg(not(all(feature = "strings", feature = "concat_str")))]
                                {
                                    None
                                }
                            },
                        }
                    },
                    Minus => eval_binary_same_type!(left_aexpr, -, right_aexpr),
                    Multiply => eval_binary_same_type!(left_aexpr, *, right_aexpr),
                    Divide => eval_binary_same_type!(left_aexpr, /, right_aexpr),
                    TrueDivide => {
                        if let (AExpr::Literal(lit_left), AExpr::Literal(lit_right)) =
                            (left_aexpr, right_aexpr)
                        {
                            match (lit_left, lit_right) {
                                (LiteralValue::Float32(x), LiteralValue::Float32(y)) => {
                                    Some(AExpr::Literal(LiteralValue::Float32(x / y)))
                                },
                                (LiteralValue::Float64(x), LiteralValue::Float64(y)) => {
                                    Some(AExpr::Literal(LiteralValue::Float64(x / y)))
                                },
                                #[cfg(feature = "dtype-i8")]
                                (LiteralValue::Int8(x), LiteralValue::Int8(y)) => Some(
                                    AExpr::Literal(LiteralValue::Float64(*x as f64 / *y as f64)),
                                ),
                                #[cfg(feature = "dtype-i16")]
                                (LiteralValue::Int16(x), LiteralValue::Int16(y)) => Some(
                                    AExpr::Literal(LiteralValue::Float64(*x as f64 / *y as f64)),
                                ),
                                (LiteralValue::Int32(x), LiteralValue::Int32(y)) => Some(
                                    AExpr::Literal(LiteralValue::Float64(*x as f64 / *y as f64)),
                                ),
                                (LiteralValue::Int64(x), LiteralValue::Int64(y)) => Some(
                                    AExpr::Literal(LiteralValue::Float64(*x as f64 / *y as f64)),
                                ),
                                #[cfg(feature = "dtype-u8")]
                                (LiteralValue::UInt8(x), LiteralValue::UInt8(y)) => Some(
                                    AExpr::Literal(LiteralValue::Float64(*x as f64 / *y as f64)),
                                ),
                                #[cfg(feature = "dtype-u16")]
                                (LiteralValue::UInt16(x), LiteralValue::UInt16(y)) => Some(
                                    AExpr::Literal(LiteralValue::Float64(*x as f64 / *y as f64)),
                                ),
                                (LiteralValue::UInt32(x), LiteralValue::UInt32(y)) => Some(
                                    AExpr::Literal(LiteralValue::Float64(*x as f64 / *y as f64)),
                                ),
                                (LiteralValue::UInt64(x), LiteralValue::UInt64(y)) => Some(
                                    AExpr::Literal(LiteralValue::Float64(*x as f64 / *y as f64)),
                                ),
                                _ => None,
                            }
                        } else {
                            None
                        }
                    },
                    Modulus => eval_binary_same_type!(left_aexpr, %, right_aexpr),
                    Lt => eval_binary_bool_type!(left_aexpr, <, right_aexpr),
                    Gt => eval_binary_bool_type!(left_aexpr, >, right_aexpr),
                    Eq | EqValidity => eval_binary_bool_type!(left_aexpr, ==, right_aexpr),
                    NotEq | NotEqValidity => eval_binary_bool_type!(left_aexpr, !=, right_aexpr),
                    GtEq => eval_binary_bool_type!(left_aexpr, >=, right_aexpr),
                    LtEq => eval_binary_bool_type!(left_aexpr, <=, right_aexpr),
                    And => eval_bitwise(left_aexpr, right_aexpr, |l, r| l & r),
                    Or => eval_bitwise(left_aexpr, right_aexpr, |l, r| l | r),
                    Xor => eval_bitwise(left_aexpr, right_aexpr, |l, r| l ^ r),
                    FloorDivide => None,
                };
                if out.is_some() {
                    return Ok(out);
                }

                // Null propagation.
                let left_is_null = matches!(left_aexpr, AExpr::Literal(LiteralValue::Null));
                let right_is_null = matches!(right_aexpr, AExpr::Literal(LiteralValue::Null));
                use Operator::*;
                match (left_is_null, op, right_is_null) {
                    // all null operation null -> null
                    (true, _, true) => Some(AExpr::Literal(LiteralValue::Null)),
                    // null == column -> column.is_null()
                    (true, Eq, false) => Some(AExpr::Function {
                        input: vec![*right],
                        function: BooleanFunction::IsNull.into(),
                        options: FunctionOptions {
                            collect_groups: ApplyOptions::GroupWise,
                            ..Default::default()
                        },
                    }),
                    // column == null -> column.is_null()
                    (false, Eq, true) => Some(AExpr::Function {
                        input: vec![*left],
                        function: BooleanFunction::IsNull.into(),
                        options: FunctionOptions {
                            collect_groups: ApplyOptions::GroupWise,
                            ..Default::default()
                        },
                    }),
                    // null != column -> column.is_not_null()
                    (true, NotEq, false) => Some(AExpr::Function {
                        input: vec![*right],
                        function: BooleanFunction::IsNotNull.into(),
                        options: FunctionOptions {
                            collect_groups: ApplyOptions::GroupWise,
                            ..Default::default()
                        },
                    }),
                    // column != null -> column.is_not_null()
                    (false, NotEq, true) => Some(AExpr::Function {
                        input: vec![*left],
                        function: BooleanFunction::IsNotNull.into(),
                        options: FunctionOptions {
                            collect_groups: ApplyOptions::GroupWise,
                            ..Default::default()
                        },
                    }),
                    _ => None,
                }
            },
            // sort().reverse() -> sort(reverse)
            // sort_by().reverse() -> sort_by(reverse)
            AExpr::Function {
                input,
                function: FunctionExpr::Reverse,
                ..
            } => {
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
                    // TODO: add support for cumsum and other operation that allow reversing.
                    _ => None,
                }
            },
            AExpr::Cast {
                expr,
                data_type,
                strict,
            } => {
                let input = expr_arena.get(*expr);
                inline_cast(input, data_type, *strict)?
            },
            // flatten nested concat_str calls
            #[cfg(all(feature = "strings", feature = "concat_str"))]
            AExpr::Function {
                input,
                function:
                    ref function @ FunctionExpr::StringExpr(StringFunction::ConcatHorizontal(sep)),
                options,
            } if sep.is_empty() => {
                if input
                    .iter()
                    .any(|node| is_string_concat(expr_arena.get(*node)))
                {
                    let mut new_inputs = Vec::with_capacity(input.len() * 2);

                    for node in input {
                        match get_string_concat_input(*node, expr_arena) {
                            Some(inp) => new_inputs.extend_from_slice(inp),
                            None => new_inputs.push(*node),
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

            _ => None,
        };
        Ok(out)
    }
}

fn inline_cast(input: &AExpr, dtype: &DataType, strict: bool) -> PolarsResult<Option<AExpr>> {
    let lv = match (input, dtype) {
        (AExpr::Literal(lv), _) if !matches!(dtype, DataType::Unknown) => match lv {
            LiteralValue::Series(s) => {
                let s = if strict {
                    s.strict_cast(dtype)
                } else {
                    s.cast(dtype)
                }?;
                LiteralValue::Series(SpecialEq::new(s))
            },
            _ => {
                let Some(av) = lv.to_anyvalue() else {
                    return Ok(None);
                };
                match (av, dtype) {
                    // casting null always remains null
                    (AnyValue::Null, _) => return Ok(None),
                    // series cast should do this one
                    #[cfg(feature = "dtype-datetime")]
                    (AnyValue::Datetime(_, _, _), DataType::Datetime(_, _)) => return Ok(None),
                    #[cfg(feature = "dtype-duration")]
                    (AnyValue::Duration(_, _), _) => return Ok(None),
                    #[cfg(feature = "dtype-categorical")]
                    (AnyValue::Categorical(_, _, _), _) | (_, DataType::Categorical(_)) => {
                        return Ok(None)
                    },
                    #[cfg(feature = "dtype-struct")]
                    (_, DataType::Struct(_)) => return Ok(None),
                    (av, _) => {
                        // raise in debug builds so we can fix them
                        // in release we continue and apply the cast later
                        #[cfg(debug_assertions)]
                        let out = { av.cast(dtype)? };
                        #[cfg(not(debug_assertions))]
                        let out = {
                            match av.cast(&dtype) {
                                Ok(out) => out,
                                Err(_) => return Ok(None),
                            }
                        };
                        out.try_into()?
                    },
                }
            },
        },
        _ => return Ok(None),
    };
    Ok(Some(AExpr::Literal(lv)))
}

#[test]
#[cfg(feature = "dtype-i8")]
fn test_expr_to_aexp() {
    use super::*;

    let expr = Expr::Literal(LiteralValue::Int8(0));
    let mut arena = Arena::new();
    let aexpr = to_aexpr(expr, &mut arena);
    assert_eq!(aexpr, Node(0));
    assert!(matches!(
        arena.get(aexpr),
        AExpr::Literal(LiteralValue::Int8(0))
    ))
}
