use polars_utils::floor_divmod::FloorDivMod;
use polars_utils::total_ord::ToTotalOrd;

use crate::plans::*;
use crate::prelude::optimizer::simplify_functions::optimize_functions;

macro_rules! eval_binary_same_type {
    ($lhs:expr, $rhs:expr, |$l: ident, $r: ident| $ret: expr) => {{
        if let (AExpr::Literal(lit_left), AExpr::Literal(lit_right)) = ($lhs, $rhs) {
            match (lit_left, lit_right) {
                (LiteralValue::Float32($l), LiteralValue::Float32($r)) => {
                    Some(AExpr::Literal(LiteralValue::Float32($ret)))
                },
                (LiteralValue::Float64($l), LiteralValue::Float64($r)) => {
                    Some(AExpr::Literal(LiteralValue::Float64($ret)))
                },
                #[cfg(feature = "dtype-i8")]
                (LiteralValue::Int8($l), LiteralValue::Int8($r)) => {
                    Some(AExpr::Literal(LiteralValue::Int8($ret)))
                },
                #[cfg(feature = "dtype-i16")]
                (LiteralValue::Int16($l), LiteralValue::Int16($r)) => {
                    Some(AExpr::Literal(LiteralValue::Int16($ret)))
                },
                (LiteralValue::Int32($l), LiteralValue::Int32($r)) => {
                    Some(AExpr::Literal(LiteralValue::Int32($ret)))
                },
                (LiteralValue::Int64($l), LiteralValue::Int64($r)) => {
                    Some(AExpr::Literal(LiteralValue::Int64($ret)))
                },
                #[cfg(feature = "dtype-u8")]
                (LiteralValue::UInt8($l), LiteralValue::UInt8($r)) => {
                    Some(AExpr::Literal(LiteralValue::UInt8($ret)))
                },
                #[cfg(feature = "dtype-u16")]
                (LiteralValue::UInt16($l), LiteralValue::UInt16($r)) => {
                    Some(AExpr::Literal(LiteralValue::UInt16($ret)))
                },
                (LiteralValue::UInt32($l), LiteralValue::UInt32($r)) => {
                    Some(AExpr::Literal(LiteralValue::UInt32($ret)))
                },
                (LiteralValue::UInt64($l), LiteralValue::UInt64($r)) => {
                    Some(AExpr::Literal(LiteralValue::UInt64($ret)))
                },
                (LiteralValue::Float($l), LiteralValue::Float($r)) => {
                    Some(AExpr::Literal(LiteralValue::Float($ret)))
                },
                (LiteralValue::Int($l), LiteralValue::Int($r)) => {
                    Some(AExpr::Literal(LiteralValue::Int($ret)))
                },
                _ => None,
            }
        } else {
            None
        }
    }};
}

macro_rules! eval_binary_cmp_same_type {
    ($lhs:expr, $operand: tt, $rhs:expr) => {{
    if let (AExpr::Literal(lit_left), AExpr::Literal(lit_right)) = ($lhs, $rhs) {
        match (lit_left, lit_right) {
            (LiteralValue::Float32(x), LiteralValue::Float32(y)) => {
                Some(AExpr::Literal(LiteralValue::Boolean(x.to_total_ord() $operand y.to_total_ord())))
            }
            (LiteralValue::Float64(x), LiteralValue::Float64(y)) => {
                Some(AExpr::Literal(LiteralValue::Boolean(x.to_total_ord() $operand y.to_total_ord())))
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
            },
            (LiteralValue::Int(x), LiteralValue::Int(y)) => {
                Some(AExpr::Literal(LiteralValue::Boolean(x $operand y)))
            }
            (LiteralValue::Float(x), LiteralValue::Float(y)) => {
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
        lp_arena: &Arena<IR>,
        lp_node: Node,
    ) -> PolarsResult<Option<AExpr>> {
        let expr = expr_arena.get(expr_node);
        let in_filter = matches!(lp_arena.get(lp_node), IR::Filter { .. });

        let out = match expr {
            // true AND x => x
            AExpr::BinaryExpr {
                left,
                op: Operator::And,
                right,
            } if matches!(
                expr_arena.get(*left),
                AExpr::Literal(LiteralValue::Boolean(true))
            ) && in_filter =>
            {
                // Only in filter as we we might change the name from "literal"
                // to whatever lhs columns is.
                return Ok(Some(expr_arena.get(*right).clone()));
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
            ) && in_filter =>
            {
                // Only in filter as we we might change the name from "literal"
                // to whatever lhs columns is.
                return Ok(Some(expr_arena.get(*right).clone()));
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
            AExpr::Function {
                input,
                function: FunctionExpr::Negate,
                ..
            } if input.len() == 1 => {
                let input = &input[0];
                let ae = expr_arena.get(input.node());
                eval_negate(ae)
            },
            _ => None,
        };
        Ok(out)
    }
}

fn eval_negate(ae: &AExpr) -> Option<AExpr> {
    let out = match ae {
        AExpr::Literal(lv) => match lv {
            #[cfg(feature = "dtype-i8")]
            LiteralValue::Int8(v) => LiteralValue::Int8(-*v),
            #[cfg(feature = "dtype-i16")]
            LiteralValue::Int16(v) => LiteralValue::Int16(-*v),
            LiteralValue::Int32(v) => LiteralValue::Int32(-*v),
            LiteralValue::Int64(v) => LiteralValue::Int64(-*v),
            LiteralValue::Float32(v) => LiteralValue::Float32(-*v),
            LiteralValue::Float64(v) => LiteralValue::Float64(-*v),
            LiteralValue::Float(v) => LiteralValue::Float(-*v),
            LiteralValue::Int(v) => LiteralValue::Int(-*v),
            _ => return None,
        },
        _ => return None,
    };
    Some(AExpr::Literal(out))
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
    lp_arena: &Arena<IR>,
    lp_node: Node,
    expr_arena: &Arena<AExpr>,
    left_node: Node,
    right_node: Node,
    left_aexpr: &AExpr,
    right_aexpr: &AExpr,
) -> Option<AExpr> {
    {
        let lp = lp_arena.get(lp_node);
        let input = lp.get_input()?;
        let schema = lp_arena.get(input).schema(lp_arena);
        let left_e = ExprIR::from_node(left_node, expr_arena);
        let right_e = ExprIR::from_node(right_node, expr_arena);

        let get_type = |ae: &AExpr| ae.get_type(&schema, Context::Default, expr_arena).ok();
        let type_a = get_type(left_aexpr).or_else(|| get_type(right_aexpr))?;
        let type_b = get_type(right_aexpr).or_else(|| get_type(right_aexpr))?;

        if type_a != type_b {
            return None;
        }

        if type_a.is_string() {
            match (left_aexpr, right_aexpr) {
                // concat + concat
                (
                    AExpr::Function {
                        input: input_left,
                        function:
                            ref fun_l @ FunctionExpr::StringExpr(StringFunction::ConcatHorizontal {
                                delimiter: sep_l,
                                ignore_nulls: ignore_nulls_l,
                            }),
                        options,
                    },
                    AExpr::Function {
                        input: input_right,
                        function:
                            FunctionExpr::StringExpr(StringFunction::ConcatHorizontal {
                                delimiter: sep_r,
                                ignore_nulls: ignore_nulls_r,
                            }),
                        ..
                    },
                ) => {
                    if sep_l.is_empty() && sep_r.is_empty() && ignore_nulls_l == ignore_nulls_r {
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
                            ref fun @ FunctionExpr::StringExpr(StringFunction::ConcatHorizontal {
                                delimiter: sep,
                                ignore_nulls,
                            }),
                        options,
                    },
                    _,
                ) => {
                    if sep.is_empty() && !ignore_nulls {
                        let mut input = input.clone();
                        input.push(right_e);
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
                            ref fun @ FunctionExpr::StringExpr(StringFunction::ConcatHorizontal {
                                delimiter: sep,
                                ignore_nulls,
                            }),
                        options,
                    },
                ) => {
                    if sep.is_empty() && !ignore_nulls {
                        let mut input = Vec::with_capacity(1 + input_right.len());
                        input.push(left_e);
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
                    input: vec![left_e, right_e],
                    function: StringFunction::ConcatHorizontal {
                        delimiter: "".to_string(),
                        ignore_nulls: false,
                    }
                    .into(),
                    options: FunctionOptions {
                        collect_groups: ApplyOptions::ElementWise,
                        flags: FunctionFlags::default()
                            | FunctionFlags::INPUT_WILDCARD_EXPANSION
                                & !FunctionFlags::RETURNS_SCALAR,
                        ..Default::default()
                    },
                }),
            }
        } else {
            None
        }
    }
}

pub struct SimplifyExprRule {}

impl OptimizationRule for SimplifyExprRule {
    #[allow(clippy::float_cmp)]
    fn optimize_expr(
        &mut self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        lp_arena: &Arena<IR>,
        lp_node: Node,
    ) -> PolarsResult<Option<AExpr>> {
        let expr = expr_arena.get(expr_node).clone();

        let out = match &expr {
            // lit(left) + lit(right) => lit(left + right)
            // and null propagation
            AExpr::BinaryExpr { left, op, right } => {
                let left_aexpr = expr_arena.get(*left);
                let right_aexpr = expr_arena.get(*right);

                // lit(left) + lit(right) => lit(left + right)
                use Operator::*;
                #[allow(clippy::manual_map)]
                let out = match op {
                    Plus => {
                        match eval_binary_same_type!(left_aexpr, right_aexpr, |l, r| l + r) {
                            Some(new) => Some(new),
                            None => {
                                // try to replace addition of string columns with `concat_str`
                                #[cfg(all(feature = "strings", feature = "concat_str"))]
                                {
                                    string_addition_to_linear_concat(
                                        lp_arena,
                                        lp_node,
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
                    Minus => eval_binary_same_type!(left_aexpr, right_aexpr, |l, r| l - r),
                    Multiply => eval_binary_same_type!(left_aexpr, right_aexpr, |l, r| l * r),
                    Divide => {
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
                                (LiteralValue::Float(x), LiteralValue::Float(y)) => {
                                    Some(AExpr::Literal(LiteralValue::Float64(x / y)))
                                },
                                #[cfg(feature = "dtype-i8")]
                                (LiteralValue::Int8(x), LiteralValue::Int8(y)) => {
                                    Some(AExpr::Literal(LiteralValue::Int8(
                                        x.wrapping_floor_div_mod(*y).0,
                                    )))
                                },
                                #[cfg(feature = "dtype-i16")]
                                (LiteralValue::Int16(x), LiteralValue::Int16(y)) => {
                                    Some(AExpr::Literal(LiteralValue::Int16(
                                        x.wrapping_floor_div_mod(*y).0,
                                    )))
                                },
                                (LiteralValue::Int32(x), LiteralValue::Int32(y)) => {
                                    Some(AExpr::Literal(LiteralValue::Int32(
                                        x.wrapping_floor_div_mod(*y).0,
                                    )))
                                },
                                (LiteralValue::Int64(x), LiteralValue::Int64(y)) => {
                                    Some(AExpr::Literal(LiteralValue::Int64(
                                        x.wrapping_floor_div_mod(*y).0,
                                    )))
                                },
                                (LiteralValue::Int(x), LiteralValue::Int(y)) => {
                                    Some(AExpr::Literal(LiteralValue::Int(
                                        x.wrapping_floor_div_mod(*y).0,
                                    )))
                                },
                                #[cfg(feature = "dtype-u8")]
                                (LiteralValue::UInt8(x), LiteralValue::UInt8(y)) => {
                                    Some(AExpr::Literal(LiteralValue::UInt8(x / y)))
                                },
                                #[cfg(feature = "dtype-u16")]
                                (LiteralValue::UInt16(x), LiteralValue::UInt16(y)) => {
                                    Some(AExpr::Literal(LiteralValue::UInt16(x / y)))
                                },
                                (LiteralValue::UInt32(x), LiteralValue::UInt32(y)) => {
                                    Some(AExpr::Literal(LiteralValue::UInt32(x / y)))
                                },
                                (LiteralValue::UInt64(x), LiteralValue::UInt64(y)) => {
                                    Some(AExpr::Literal(LiteralValue::UInt64(x / y)))
                                },
                                _ => None,
                            }
                        } else {
                            None
                        }
                    },
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
                                (LiteralValue::Float(x), LiteralValue::Float(y)) => {
                                    Some(AExpr::Literal(LiteralValue::Float(x / y)))
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
                                (LiteralValue::Int(x), LiteralValue::Int(y)) => {
                                    Some(AExpr::Literal(LiteralValue::Float(*x as f64 / *y as f64)))
                                },
                                _ => None,
                            }
                        } else {
                            None
                        }
                    },
                    Modulus => eval_binary_same_type!(left_aexpr, right_aexpr, |l, r| l
                        .wrapping_floor_div_mod(*r)
                        .1),
                    Lt => eval_binary_cmp_same_type!(left_aexpr, <, right_aexpr),
                    Gt => eval_binary_cmp_same_type!(left_aexpr, >, right_aexpr),
                    Eq | EqValidity => eval_binary_cmp_same_type!(left_aexpr, ==, right_aexpr),
                    NotEq | NotEqValidity => {
                        eval_binary_cmp_same_type!(left_aexpr, !=, right_aexpr)
                    },
                    GtEq => eval_binary_cmp_same_type!(left_aexpr, >=, right_aexpr),
                    LtEq => eval_binary_cmp_same_type!(left_aexpr, <=, right_aexpr),
                    And | LogicalAnd => eval_bitwise(left_aexpr, right_aexpr, |l, r| l & r),
                    Or | LogicalOr => eval_bitwise(left_aexpr, right_aexpr, |l, r| l | r),
                    Xor => eval_bitwise(left_aexpr, right_aexpr, |l, r| l ^ r),
                    FloorDivide => eval_binary_same_type!(left_aexpr, right_aexpr, |l, r| l
                        .wrapping_floor_div_mod(*r)
                        .0),
                };
                if out.is_some() {
                    return Ok(out);
                }

                None
            },
            AExpr::Function {
                input,
                function,
                options,
                ..
            } => return optimize_functions(input, function, options, expr_arena),
            _ => None,
        };
        Ok(out)
    }
}

#[test]
#[cfg(feature = "dtype-i8")]
fn test_expr_to_aexp() {
    use super::*;

    let expr = Expr::Literal(LiteralValue::Int8(0));
    let mut arena = Arena::new();
    let aexpr = to_aexpr(expr, &mut arena).unwrap();
    assert_eq!(aexpr, Node(0));
    assert!(matches!(
        arena.get(aexpr),
        AExpr::Literal(LiteralValue::Int8(0))
    ))
}
