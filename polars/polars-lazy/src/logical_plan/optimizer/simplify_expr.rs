use polars_utils::arena::Arena;

use crate::logical_plan::optimizer::stack_opt::OptimizationRule;
use crate::logical_plan::*;

macro_rules! eval_binary_same_type {
    ($lhs:expr, $operand: tt, $rhs:expr) => {{
    if let (AExpr::Literal(lit_left), AExpr::Literal(lit_right)) = ($lhs, $rhs) {
        return match (lit_left, lit_right) {
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
        };
    }
    None

    }}
}

macro_rules! eval_binary_bool_type {
    ($lhs:expr, $operand: tt, $rhs:expr) => {{
    if let (AExpr::Literal(lit_left), AExpr::Literal(lit_right)) = ($lhs, $rhs) {
        return match (lit_left, lit_right) {
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
        };
    }
    None

    }}
}

pub(crate) struct SimplifyBooleanRule {}

impl OptimizationRule for SimplifyBooleanRule {
    fn optimize_expr(
        &self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        _: &Arena<ALogicalPlan>,
        _: Node,
    ) -> Option<AExpr> {
        let expr = expr_arena.get(expr_node);
        match expr {
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
                Some(expr_arena.get(*right).clone())
            }
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
            }
            // x AND false -> false
            AExpr::BinaryExpr {
                op: Operator::And,
                right,
                ..
            } if matches!(
                expr_arena.get(*right),
                AExpr::Literal(LiteralValue::Boolean(false))
            ) =>
            {
                Some(AExpr::Literal(LiteralValue::Boolean(false)))
            }
            // false AND x -> false
            AExpr::BinaryExpr {
                left,
                op: Operator::And,
                ..
            } if matches!(
                expr_arena.get(*left),
                AExpr::Literal(LiteralValue::Boolean(false))
            ) =>
            {
                Some(AExpr::Literal(LiteralValue::Boolean(false)))
            }
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
                Some(expr_arena.get(*right).clone())
            }
            // x or false => x
            AExpr::BinaryExpr {
                op: Operator::Or,
                right,
                ..
            } if matches!(
                expr_arena.get(*right),
                AExpr::Literal(LiteralValue::Boolean(false))
            ) =>
            {
                Some(expr_arena.get(*right).clone())
            }

            // false OR x => x
            AExpr::BinaryExpr {
                left,
                op: Operator::Or,
                right,
            } if matches!(
                expr_arena.get(*left),
                AExpr::Literal(LiteralValue::Boolean(false))
            ) =>
            {
                Some(expr_arena.get(*right).clone())
            }

            // true OR x => true
            AExpr::BinaryExpr {
                op: Operator::Or,
                right,
                ..
            } if matches!(
                expr_arena.get(*right),
                AExpr::Literal(LiteralValue::Boolean(true))
            ) =>
            {
                Some(AExpr::Literal(LiteralValue::Boolean(false)))
            }

            // x OR true => true
            AExpr::BinaryExpr {
                op: Operator::Or,
                left,
                ..
            } if matches!(
                expr_arena.get(*left),
                AExpr::Literal(LiteralValue::Boolean(true))
            ) =>
            {
                Some(AExpr::Literal(LiteralValue::Boolean(false)))
            }

            AExpr::Not(x) => {
                let y = expr_arena.get(*x);

                match y {
                    // not(not x) => x
                    AExpr::Not(expr) => Some(expr_arena.get(*expr).clone()),
                    // not(lit x) => !x
                    AExpr::Literal(LiteralValue::Boolean(b)) => {
                        Some(AExpr::Literal(LiteralValue::Boolean(!b)))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
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
            }
            _ => None,
        };
    }
    None
}

pub struct SimplifyExprRule {}

impl OptimizationRule for SimplifyExprRule {
    #[allow(clippy::float_cmp)]
    fn optimize_expr(
        &self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        _: &Arena<ALogicalPlan>,
        _: Node,
    ) -> Option<AExpr> {
        let expr = expr_arena.get(expr_node);
        match expr {
            // lit(left) + lit(right) => lit(left + right)
            // and null propagation
            AExpr::BinaryExpr { left, op, right } => {
                let left_aexpr = expr_arena.get(*left);
                let right_aexpr = expr_arena.get(*right);

                // lit(left) + lit(right) => lit(left + right)
                let out = match op {
                    Operator::Plus => eval_binary_same_type!(left_aexpr, +, right_aexpr),
                    Operator::Minus => eval_binary_same_type!(left_aexpr, -, right_aexpr),
                    Operator::Multiply => eval_binary_same_type!(left_aexpr, *, right_aexpr),
                    Operator::Divide => eval_binary_same_type!(left_aexpr, /, right_aexpr),
                    Operator::TrueDivide => {
                        if let (AExpr::Literal(lit_left), AExpr::Literal(lit_right)) =
                            (left_aexpr, right_aexpr)
                        {
                            return match (lit_left, lit_right) {
                                (LiteralValue::Float32(x), LiteralValue::Float32(y)) => {
                                    Some(AExpr::Literal(LiteralValue::Float32(x / y)))
                                }
                                (LiteralValue::Float64(x), LiteralValue::Float64(y)) => {
                                    Some(AExpr::Literal(LiteralValue::Float64(x / y)))
                                }
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
                            };
                        }
                        None
                    }
                    Operator::Modulus => eval_binary_same_type!(left_aexpr, %, right_aexpr),
                    Operator::Lt => eval_binary_bool_type!(left_aexpr, <, right_aexpr),
                    Operator::Gt => eval_binary_bool_type!(left_aexpr, >, right_aexpr),
                    Operator::Eq => eval_binary_bool_type!(left_aexpr, ==, right_aexpr),
                    Operator::NotEq => eval_binary_bool_type!(left_aexpr, !=, right_aexpr),
                    Operator::GtEq => eval_binary_bool_type!(left_aexpr, >=, right_aexpr),
                    Operator::LtEq => eval_binary_bool_type!(left_aexpr, >=, right_aexpr),
                    Operator::And => eval_bitwise(left_aexpr, right_aexpr, |l, r| l & r),
                    Operator::Or => eval_bitwise(left_aexpr, right_aexpr, |l, r| l | r),
                    Operator::Xor => eval_bitwise(left_aexpr, right_aexpr, |l, r| l ^ r),
                };
                if out.is_some() {
                    return out;
                }

                // Null propagation.
                let left_is_null = matches!(left_aexpr, AExpr::Literal(LiteralValue::Null));
                let right_is_null = matches!(right_aexpr, AExpr::Literal(LiteralValue::Null));
                use Operator::*;
                match (left_is_null, op, right_is_null) {
                    // all null operation null -> null
                    (true, _, true) => Some(AExpr::Literal(LiteralValue::Null)),
                    // null == column -> column.is_null()
                    (true, Eq, false) => Some(AExpr::IsNull(*right)),
                    // column == null -> column.is_null()
                    (false, Eq, true) => Some(AExpr::IsNull(*left)),
                    // null != column -> column.is_not_null()
                    (true, NotEq, false) => Some(AExpr::IsNotNull(*right)),
                    // column != null -> column.is_not_null()
                    (false, NotEq, true) => Some(AExpr::IsNotNull(*left)),
                    _ => None,
                }
            }
            AExpr::Reverse(expr) => {
                let input = expr_arena.get(*expr);
                match input {
                    AExpr::Sort { expr, options } => {
                        let mut options = *options;
                        options.descending = !options.descending;
                        Some(AExpr::Sort {
                            expr: *expr,
                            options,
                        })
                    }
                    AExpr::SortBy { expr, by, reverse } => Some(AExpr::SortBy {
                        expr: *expr,
                        by: by.clone(),
                        reverse: reverse.iter().map(|r| !*r).collect(),
                    }),
                    // TODO: add support for cumsum and other operation that allow reversing.
                    _ => None,
                }
            }
            AExpr::Cast {
                expr, data_type, ..
            } => {
                let input = expr_arena.get(*expr);
                // faster casts (we only do strict casts)
                match (input, data_type) {
                    #[cfg(feature = "dtype-i8")]
                    (AExpr::Literal(LiteralValue::Int8(v)), DataType::Int64) => {
                        Some(AExpr::Literal(LiteralValue::Int64(*v as i64)))
                    }
                    #[cfg(feature = "dtype-i16")]
                    (AExpr::Literal(LiteralValue::Int16(v)), DataType::Int64) => {
                        Some(AExpr::Literal(LiteralValue::Int64(*v as i64)))
                    }
                    (AExpr::Literal(LiteralValue::Int32(v)), DataType::Int64) => {
                        Some(AExpr::Literal(LiteralValue::Int64(*v as i64)))
                    }
                    (AExpr::Literal(LiteralValue::UInt32(v)), DataType::Int64) => {
                        Some(AExpr::Literal(LiteralValue::Int64(*v as i64)))
                    }
                    (AExpr::Literal(LiteralValue::Float32(v)), DataType::Float64) => {
                        Some(AExpr::Literal(LiteralValue::Float64(*v as f64)))
                    }

                    #[cfg(feature = "dtype-i16")]
                    (AExpr::Literal(LiteralValue::Int8(v)), DataType::Float64) => {
                        Some(AExpr::Literal(LiteralValue::Float64(*v as f64)))
                    }
                    #[cfg(feature = "dtype-i16")]
                    (AExpr::Literal(LiteralValue::Int16(v)), DataType::Float64) => {
                        Some(AExpr::Literal(LiteralValue::Float64(*v as f64)))
                    }
                    (AExpr::Literal(LiteralValue::Int32(v)), DataType::Float64) => {
                        Some(AExpr::Literal(LiteralValue::Float64(*v as f64)))
                    }
                    (AExpr::Literal(LiteralValue::Int64(v)), DataType::Float64) => {
                        Some(AExpr::Literal(LiteralValue::Float64(*v as f64)))
                    }
                    #[cfg(feature = "dtype-u8")]
                    (AExpr::Literal(LiteralValue::UInt8(v)), DataType::Float64) => {
                        Some(AExpr::Literal(LiteralValue::Float64(*v as f64)))
                    }
                    #[cfg(feature = "dtype-u16")]
                    (AExpr::Literal(LiteralValue::UInt16(v)), DataType::Float64) => {
                        Some(AExpr::Literal(LiteralValue::Float64(*v as f64)))
                    }
                    (AExpr::Literal(LiteralValue::UInt32(v)), DataType::Float64) => {
                        Some(AExpr::Literal(LiteralValue::Float64(*v as f64)))
                    }
                    (AExpr::Literal(LiteralValue::UInt64(v)), DataType::Float64) => {
                        Some(AExpr::Literal(LiteralValue::Float64(*v as f64)))
                    }

                    #[cfg(feature = "dtype-i16")]
                    (AExpr::Literal(LiteralValue::Int8(v)), DataType::Float32) => {
                        Some(AExpr::Literal(LiteralValue::Float32(*v as f32)))
                    }
                    #[cfg(feature = "dtype-i16")]
                    (AExpr::Literal(LiteralValue::Int16(v)), DataType::Float32) => {
                        Some(AExpr::Literal(LiteralValue::Float32(*v as f32)))
                    }
                    (AExpr::Literal(LiteralValue::Int32(v)), DataType::Float32) => {
                        Some(AExpr::Literal(LiteralValue::Float32(*v as f32)))
                    }
                    (AExpr::Literal(LiteralValue::Int64(v)), DataType::Float32) => {
                        Some(AExpr::Literal(LiteralValue::Float32(*v as f32)))
                    }
                    #[cfg(feature = "dtype-u8")]
                    (AExpr::Literal(LiteralValue::UInt8(v)), DataType::Float32) => {
                        Some(AExpr::Literal(LiteralValue::Float32(*v as f32)))
                    }
                    #[cfg(feature = "dtype-u16")]
                    (AExpr::Literal(LiteralValue::UInt16(v)), DataType::Float32) => {
                        Some(AExpr::Literal(LiteralValue::Float32(*v as f32)))
                    }
                    (AExpr::Literal(LiteralValue::UInt32(v)), DataType::Float32) => {
                        Some(AExpr::Literal(LiteralValue::Float32(*v as f32)))
                    }
                    (AExpr::Literal(LiteralValue::UInt64(v)), DataType::Float32) => {
                        Some(AExpr::Literal(LiteralValue::Float32(*v as f32)))
                    }
                    _ => None,
                }
            }

            _ => None,
        }
    }
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
