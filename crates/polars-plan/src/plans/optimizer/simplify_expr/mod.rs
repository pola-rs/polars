mod simplify_functions;

use polars_utils::floor_divmod::FloorDivMod;
use polars_utils::total_ord::ToTotalOrd;
use simplify_functions::optimize_functions;

use crate::plans::*;

fn new_null_count(input: &[ExprIR]) -> AExpr {
    AExpr::Function {
        input: input.to_vec(),
        function: FunctionExpr::NullCount,
        options: FunctionOptions::aggregation()
            .with_flags(|f| f | FunctionFlags::ALLOW_GROUP_AWARE),
    }
}

macro_rules! eval_binary_same_type {
    ($lhs:expr, $rhs:expr, |$l: ident, $r: ident| $ret: expr) => {{
        if let (AExpr::Literal(lit_left), AExpr::Literal(lit_right)) = ($lhs, $rhs) {
            match (lit_left, lit_right) {
                (LiteralValue::Scalar(l), LiteralValue::Scalar(r)) => {
                    match (l.as_any_value(), r.as_any_value()) {
                        (AnyValue::Float32($l), AnyValue::Float32($r)) => {
                            Some(AExpr::Literal(<Scalar as From<f32>>::from($ret).into()))
                        },
                        (AnyValue::Float64($l), AnyValue::Float64($r)) => {
                            Some(AExpr::Literal(<Scalar as From<f64>>::from($ret).into()))
                        },

                        (AnyValue::Int8($l), AnyValue::Int8($r)) => {
                            Some(AExpr::Literal(<Scalar as From<i8>>::from($ret).into()))
                        },
                        (AnyValue::Int16($l), AnyValue::Int16($r)) => {
                            Some(AExpr::Literal(<Scalar as From<i16>>::from($ret).into()))
                        },
                        (AnyValue::Int32($l), AnyValue::Int32($r)) => {
                            Some(AExpr::Literal(<Scalar as From<i32>>::from($ret).into()))
                        },
                        (AnyValue::Int64($l), AnyValue::Int64($r)) => {
                            Some(AExpr::Literal(<Scalar as From<i64>>::from($ret).into()))
                        },
                        (AnyValue::Int128($l), AnyValue::Int128($r)) => {
                            Some(AExpr::Literal(<Scalar as From<i128>>::from($ret).into()))
                        },

                        (AnyValue::UInt8($l), AnyValue::UInt8($r)) => {
                            Some(AExpr::Literal(<Scalar as From<u8>>::from($ret).into()))
                        },
                        (AnyValue::UInt16($l), AnyValue::UInt16($r)) => {
                            Some(AExpr::Literal(<Scalar as From<u16>>::from($ret).into()))
                        },
                        (AnyValue::UInt32($l), AnyValue::UInt32($r)) => {
                            Some(AExpr::Literal(<Scalar as From<u32>>::from($ret).into()))
                        },
                        (AnyValue::UInt64($l), AnyValue::UInt64($r)) => {
                            Some(AExpr::Literal(<Scalar as From<u64>>::from($ret).into()))
                        },

                        _ => None,
                    }
                    .into()
                },
                (
                    LiteralValue::Dyn(DynLiteralValue::Float($l)),
                    LiteralValue::Dyn(DynLiteralValue::Float($r)),
                ) => {
                    let $l = *$l;
                    let $r = *$r;
                    Some(AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::Float(
                        $ret,
                    ))))
                },
                (
                    LiteralValue::Dyn(DynLiteralValue::Int($l)),
                    LiteralValue::Dyn(DynLiteralValue::Int($r)),
                ) => {
                    let $l = *$l;
                    let $r = *$r;
                    Some(AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(
                        $ret,
                    ))))
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
            (LiteralValue::Scalar(l), LiteralValue::Scalar(r)) => match (l.as_any_value(), r.as_any_value()) {
                (AnyValue::Float32(l), AnyValue::Float32(r)) => Some(AExpr::Literal({ let x: bool = l.to_total_ord() $operand r.to_total_ord(); Scalar::from(x) }.into())),
                (AnyValue::Float64(l), AnyValue::Float64(r)) => Some(AExpr::Literal({ let x: bool = l.to_total_ord() $operand r.to_total_ord(); Scalar::from(x) }.into())),

                (AnyValue::Boolean(l), AnyValue::Boolean(r)) => Some(AExpr::Literal({ let x: bool = l $operand r; Scalar::from(x) }.into())),

                (AnyValue::Int8(l), AnyValue::Int8(r)) => Some(AExpr::Literal({ let x: bool = l $operand r; Scalar::from(x) }.into())),
                (AnyValue::Int16(l), AnyValue::Int16(r)) => Some(AExpr::Literal({ let x: bool = l $operand r; Scalar::from(x) }.into())),
                (AnyValue::Int32(l), AnyValue::Int32(r)) => Some(AExpr::Literal({ let x: bool = l $operand r; Scalar::from(x) }.into())),
                (AnyValue::Int64(l), AnyValue::Int64(r)) => Some(AExpr::Literal({ let x: bool = l $operand r; Scalar::from(x) }.into())),
                (AnyValue::Int128(l), AnyValue::Int128(r)) => Some(AExpr::Literal({ let x: bool = l $operand r; Scalar::from(x) }.into())),

                (AnyValue::UInt8(l), AnyValue::UInt8(r)) => Some(AExpr::Literal({ let x: bool = l $operand r; Scalar::from(x) }.into())),
                (AnyValue::UInt16(l), AnyValue::UInt16(r)) => Some(AExpr::Literal({ let x: bool = l $operand r; Scalar::from(x) }.into())),
                (AnyValue::UInt32(l), AnyValue::UInt32(r)) => Some(AExpr::Literal({ let x: bool = l $operand r; Scalar::from(x) }.into())),
                (AnyValue::UInt64(l), AnyValue::UInt64(r)) => Some(AExpr::Literal({ let x: bool = l $operand r; Scalar::from(x) }.into())),

                _ => None,
            }.into(),
            (LiteralValue::Dyn(DynLiteralValue::Float(l)), LiteralValue::Dyn(DynLiteralValue::Float(r))) => {
                let x: bool = l.to_total_ord() $operand r.to_total_ord();
                Some(AExpr::Literal(Scalar::from(x).into()))
            },
            (LiteralValue::Dyn(DynLiteralValue::Int(l)), LiteralValue::Dyn(DynLiteralValue::Int(r))) => {
                let x: bool = l $operand r;
                Some(AExpr::Literal(Scalar::from(x).into()))
            },
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
                AExpr::Literal(lv) if lv.bool() == Some(true)
            ) && in_filter =>
            {
                // Only in filter as we might change the name from "literal"
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
                AExpr::Literal(lv) if lv.bool() == Some(true)
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
                    AExpr::Literal(lv) if lv.bool() == Some(false)
                ) =>
            {
                Some(AExpr::Literal(Scalar::from(false).into()))
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
                AExpr::Literal(lv) if lv.bool() == Some(false)
            ) && matches!(expr_arena.get(*right), AExpr::Literal(_)) =>
            {
                Some(AExpr::Literal(Scalar::from(false).into()))
            },

            // false or x => x
            AExpr::BinaryExpr {
                left,
                op: Operator::Or,
                right,
            } if matches!(
                expr_arena.get(*left),
                AExpr::Literal(lv) if lv.bool() == Some(false)
            ) && in_filter =>
            {
                // Only in filter as we might change the name from "literal"
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
                AExpr::Literal(lv) if lv.bool() == Some(false)
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
                    AExpr::Literal(lv) if lv.bool() == Some(true)
                ) =>
            {
                Some(AExpr::Literal(Scalar::from(true).into()))
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
                    AExpr::Literal(lv) if lv.bool() == Some(true)
            ) && matches!(expr_arena.get(*right), AExpr::Literal(_)) =>
            {
                Some(AExpr::Literal(Scalar::from(true).into()))
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
    use std::ops::Neg;
    let out = match ae {
        AExpr::Literal(lv) => match lv {
            LiteralValue::Scalar(sc) => match sc.as_any_value() {
                AnyValue::Int8(v) => Scalar::from(v.checked_neg()?),
                AnyValue::Int16(v) => Scalar::from(v.checked_neg()?),
                AnyValue::Int32(v) => Scalar::from(v.checked_neg()?),
                AnyValue::Int64(v) => Scalar::from(v.checked_neg()?),
                AnyValue::Float32(v) => Scalar::from(v.neg()),
                AnyValue::Float64(v) => Scalar::from(v.neg()),
                _ => return None,
            }
            .into(),
            LiteralValue::Dyn(d) => LiteralValue::Dyn(match d {
                DynLiteralValue::Int(v) => DynLiteralValue::Int(v.checked_neg()?),
                DynLiteralValue::Float(v) => DynLiteralValue::Float(v.neg()),
                _ => return None,
            }),
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
        return match (lit_left.bool(), lit_right.bool()) {
            (Some(x), Some(y)) => Some(AExpr::Literal(Scalar::from(operation(x, y)).into())),
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
                            fun_l @ FunctionExpr::StringExpr(StringFunction::ConcatHorizontal {
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
                            fun @ FunctionExpr::StringExpr(StringFunction::ConcatHorizontal {
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
                            fun @ FunctionExpr::StringExpr(StringFunction::ConcatHorizontal {
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
                _ => {
                    let function = StringFunction::ConcatHorizontal {
                        delimiter: "".into(),
                        ignore_nulls: false,
                    };
                    let options = function.function_options();
                    Some(AExpr::Function {
                        input: vec![left_e, right_e],
                        function: function.into(),
                        options,
                    })
                },
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
            // drop_nulls().len() -> len() - null_count()
            // drop_nulls().count() -> len() - null_count()
            AExpr::Agg(IRAggExpr::Count(input, _)) => {
                let input_expr = expr_arena.get(*input);
                match input_expr {
                    AExpr::Function {
                        input,
                        function: FunctionExpr::DropNulls,
                        options: _,
                    } => {
                        // we should perform optimization only if the original expression is a column
                        // so in case of disabled CSE, we will not suffer from performance regression
                        if input.len() == 1 {
                            let drop_nulls_input_node = input[0].node();
                            match expr_arena.get(drop_nulls_input_node) {
                                AExpr::Column(_) => Some(AExpr::BinaryExpr {
                                    op: Operator::Minus,
                                    right: expr_arena.add(new_null_count(input)),
                                    left: expr_arena.add(AExpr::Agg(IRAggExpr::Count(
                                        drop_nulls_input_node,
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
            // is_null().sum() -> null_count()
            // is_not_null().sum() -> len() - null_count()
            AExpr::Agg(IRAggExpr::Sum(input)) => {
                let input_expr = expr_arena.get(*input);
                match input_expr {
                    AExpr::Function {
                        input,
                        function: FunctionExpr::Boolean(BooleanFunction::IsNull),
                        options: _,
                    } => Some(new_null_count(input)),
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
                                    op: Operator::Minus,
                                    right: expr_arena.add(new_null_count(input)),
                                    left: expr_arena.add(AExpr::Agg(IRAggExpr::Count(
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
                                (LiteralValue::Scalar(l), LiteralValue::Scalar(r)) => {
                                    match (l.as_any_value(), r.as_any_value()) {
                                        (AnyValue::Float32(x), AnyValue::Float32(y)) => {
                                            Some(AExpr::Literal(
                                                <Scalar as From<f32>>::from(x / y).into(),
                                            ))
                                        },
                                        (AnyValue::Float64(x), AnyValue::Float64(y)) => {
                                            Some(AExpr::Literal(
                                                <Scalar as From<f64>>::from(x / y).into(),
                                            ))
                                        },

                                        (AnyValue::Int8(x), AnyValue::Int8(y)) => {
                                            Some(AExpr::Literal(
                                                <Scalar as From<i8>>::from(
                                                    x.wrapping_floor_div_mod(y).0,
                                                )
                                                .into(),
                                            ))
                                        },
                                        (AnyValue::Int16(x), AnyValue::Int16(y)) => {
                                            Some(AExpr::Literal(
                                                <Scalar as From<i16>>::from(
                                                    x.wrapping_floor_div_mod(y).0,
                                                )
                                                .into(),
                                            ))
                                        },
                                        (AnyValue::Int32(x), AnyValue::Int32(y)) => {
                                            Some(AExpr::Literal(
                                                <Scalar as From<i32>>::from(
                                                    x.wrapping_floor_div_mod(y).0,
                                                )
                                                .into(),
                                            ))
                                        },
                                        (AnyValue::Int64(x), AnyValue::Int64(y)) => {
                                            Some(AExpr::Literal(
                                                <Scalar as From<i64>>::from(
                                                    x.wrapping_floor_div_mod(y).0,
                                                )
                                                .into(),
                                            ))
                                        },
                                        (AnyValue::Int128(x), AnyValue::Int128(y)) => {
                                            Some(AExpr::Literal(
                                                <Scalar as From<i128>>::from(
                                                    x.wrapping_floor_div_mod(y).0,
                                                )
                                                .into(),
                                            ))
                                        },

                                        (AnyValue::UInt8(x), AnyValue::UInt8(y)) => {
                                            Some(AExpr::Literal(
                                                <Scalar as From<u8>>::from(x / y).into(),
                                            ))
                                        },
                                        (AnyValue::UInt16(x), AnyValue::UInt16(y)) => {
                                            Some(AExpr::Literal(
                                                <Scalar as From<u16>>::from(x / y).into(),
                                            ))
                                        },
                                        (AnyValue::UInt32(x), AnyValue::UInt32(y)) => {
                                            Some(AExpr::Literal(
                                                <Scalar as From<u32>>::from(x / y).into(),
                                            ))
                                        },
                                        (AnyValue::UInt64(x), AnyValue::UInt64(y)) => {
                                            Some(AExpr::Literal(
                                                <Scalar as From<u64>>::from(x / y).into(),
                                            ))
                                        },

                                        _ => None,
                                    }
                                },

                                (
                                    LiteralValue::Dyn(DynLiteralValue::Float(x)),
                                    LiteralValue::Dyn(DynLiteralValue::Float(y)),
                                ) => {
                                    Some(AExpr::Literal(<Scalar as From<f64>>::from(x / y).into()))
                                },
                                (
                                    LiteralValue::Dyn(DynLiteralValue::Int(x)),
                                    LiteralValue::Dyn(DynLiteralValue::Int(y)),
                                ) => Some(AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(
                                    x.wrapping_floor_div_mod(*y).0,
                                )))),
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
                                (LiteralValue::Scalar(l), LiteralValue::Scalar(r)) => {
                                    match (l.as_any_value(), r.as_any_value()) {
                                        (AnyValue::Float32(x), AnyValue::Float32(y)) => {
                                            Some(AExpr::Literal(Scalar::from(x / y).into()))
                                        },
                                        (AnyValue::Float64(x), AnyValue::Float64(y)) => {
                                            Some(AExpr::Literal(Scalar::from(x / y).into()))
                                        },

                                        (AnyValue::Int8(x), AnyValue::Int8(y)) => {
                                            Some(AExpr::Literal(
                                                Scalar::from(x as f64 / y as f64).into(),
                                            ))
                                        },
                                        (AnyValue::Int16(x), AnyValue::Int16(y)) => {
                                            Some(AExpr::Literal(
                                                Scalar::from(x as f64 / y as f64).into(),
                                            ))
                                        },
                                        (AnyValue::Int32(x), AnyValue::Int32(y)) => {
                                            Some(AExpr::Literal(
                                                Scalar::from(x as f64 / y as f64).into(),
                                            ))
                                        },
                                        (AnyValue::Int64(x), AnyValue::Int64(y)) => {
                                            Some(AExpr::Literal(
                                                Scalar::from(x as f64 / y as f64).into(),
                                            ))
                                        },
                                        (AnyValue::Int128(x), AnyValue::Int128(y)) => {
                                            Some(AExpr::Literal(
                                                Scalar::from(x as f64 / y as f64).into(),
                                            ))
                                        },

                                        (AnyValue::UInt8(x), AnyValue::UInt8(y)) => {
                                            Some(AExpr::Literal(
                                                Scalar::from(x as f64 / y as f64).into(),
                                            ))
                                        },
                                        (AnyValue::UInt16(x), AnyValue::UInt16(y)) => {
                                            Some(AExpr::Literal(
                                                Scalar::from(x as f64 / y as f64).into(),
                                            ))
                                        },
                                        (AnyValue::UInt32(x), AnyValue::UInt32(y)) => {
                                            Some(AExpr::Literal(
                                                Scalar::from(x as f64 / y as f64).into(),
                                            ))
                                        },
                                        (AnyValue::UInt64(x), AnyValue::UInt64(y)) => {
                                            Some(AExpr::Literal(
                                                Scalar::from(x as f64 / y as f64).into(),
                                            ))
                                        },

                                        _ => None,
                                    }
                                },

                                (
                                    LiteralValue::Dyn(DynLiteralValue::Float(x)),
                                    LiteralValue::Dyn(DynLiteralValue::Float(y)),
                                ) => Some(AExpr::Literal(Scalar::from(*x / *y).into())),
                                (
                                    LiteralValue::Dyn(DynLiteralValue::Int(x)),
                                    LiteralValue::Dyn(DynLiteralValue::Int(y)),
                                ) => {
                                    Some(AExpr::Literal(Scalar::from(*x as f64 / *y as f64).into()))
                                },
                                _ => None,
                            }
                        } else {
                            None
                        }
                    },
                    Modulus => eval_binary_same_type!(left_aexpr, right_aexpr, |l, r| l
                        .wrapping_floor_div_mod(r)
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
                        .wrapping_floor_div_mod(r)
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
