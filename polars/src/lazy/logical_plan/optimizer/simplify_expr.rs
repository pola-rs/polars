use crate::lazy::logical_plan::*;
use crate::lazy::prelude::*;
use crate::prelude::*;
use crate::utils::Arena;

macro_rules! eval_binary_same_type {
    ($lhs:expr, $operand: tt, $rhs:expr) => {{
    if let (AExpr::Literal(lit_left), AExpr::Literal(lit_right)) = ($lhs, $rhs) {
        return match (lit_left, lit_right) {
            (ScalarValue::Float32(x), ScalarValue::Float32(y)) => {
                Some(AExpr::Literal(ScalarValue::Float32(x $operand y)))
            }
            (ScalarValue::Float64(x), ScalarValue::Float64(y)) => {
                Some(AExpr::Literal(ScalarValue::Float64(x $operand y)))
            }
            (ScalarValue::Int8(x), ScalarValue::Int8(y)) => {
                Some(AExpr::Literal(ScalarValue::Int8(x $operand y)))
            }
            (ScalarValue::Int16(x), ScalarValue::Int16(y)) => {
                Some(AExpr::Literal(ScalarValue::Int16(x $operand y)))
            }
            (ScalarValue::Int32(x), ScalarValue::Int32(y)) => {
                Some(AExpr::Literal(ScalarValue::Int32(x $operand y)))
            }
            (ScalarValue::Int64(x), ScalarValue::Int64(y)) => {
                Some(AExpr::Literal(ScalarValue::Int64(x $operand y)))
            }
            (ScalarValue::UInt8(x), ScalarValue::UInt8(y)) => {
                Some(AExpr::Literal(ScalarValue::UInt8(x $operand y)))
            }
            (ScalarValue::UInt16(x), ScalarValue::UInt16(y)) => {
                Some(AExpr::Literal(ScalarValue::UInt16(x $operand y)))
            }
            (ScalarValue::UInt32(x), ScalarValue::UInt32(y)) => {
                Some(AExpr::Literal(ScalarValue::UInt32(x $operand y)))
            }
            (ScalarValue::UInt64(x), ScalarValue::UInt64(y)) => {
                Some(AExpr::Literal(ScalarValue::UInt64(x $operand y)))
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
            (ScalarValue::Float32(x), ScalarValue::Float32(y)) => {
                Some(AExpr::Literal(ScalarValue::Boolean(x $operand y)))
            }
            (ScalarValue::Float64(x), ScalarValue::Float64(y)) => {
                Some(AExpr::Literal(ScalarValue::Boolean(x $operand y)))
            }
            (ScalarValue::Int8(x), ScalarValue::Int8(y)) => {
                Some(AExpr::Literal(ScalarValue::Boolean(x $operand y)))
            }
            (ScalarValue::Int16(x), ScalarValue::Int16(y)) => {
                Some(AExpr::Literal(ScalarValue::Boolean(x $operand y)))
            }
            (ScalarValue::Int32(x), ScalarValue::Int32(y)) => {
                Some(AExpr::Literal(ScalarValue::Boolean(x $operand y)))
            }
            (ScalarValue::Int64(x), ScalarValue::Int64(y)) => {
                Some(AExpr::Literal(ScalarValue::Boolean(x $operand y)))
            }
            (ScalarValue::UInt8(x), ScalarValue::UInt8(y)) => {
                Some(AExpr::Literal(ScalarValue::Boolean(x $operand y)))
            }
            (ScalarValue::UInt16(x), ScalarValue::UInt16(y)) => {
                Some(AExpr::Literal(ScalarValue::Boolean(x $operand y)))
            }
            (ScalarValue::UInt32(x), ScalarValue::UInt32(y)) => {
                Some(AExpr::Literal(ScalarValue::Boolean(x $operand y)))
            }
            (ScalarValue::UInt64(x), ScalarValue::UInt64(y)) => {
                Some(AExpr::Literal(ScalarValue::Boolean(x $operand y)))
            }
            (ScalarValue::Boolean(x), ScalarValue::Boolean(y)) => {
                Some(AExpr::Literal(ScalarValue::Boolean(x $operand y)))
            }
            _ => None,
        };
    }
    None

    }}
}

pub struct SimplifyExpr {}

struct SimplifyBooleanRule {}

impl Rule for SimplifyBooleanRule {
    fn optimize_expr(&self, arena: &Arena<AExpr>, expr: &AExpr) -> Option<AExpr> {
        match expr {
            // true AND x => x
            AExpr::BinaryExpr {
                left,
                op: Operator::And,
                right,
            } if matches!(arena.get(*left), AExpr::Literal(ScalarValue::Boolean(true))) => {
                Some(arena.get(*right).clone())
            }
            // x AND true => x
            AExpr::BinaryExpr {
                left,
                op: Operator::And,
                right,
            } if matches!(
                arena.get(*right),
                AExpr::Literal(ScalarValue::Boolean(true))
            ) =>
            {
                Some(arena.get(*left).clone())
            }
            // x AND false -> false
            AExpr::BinaryExpr {
                op: Operator::And,
                right,
                ..
            } if matches!(
                arena.get(*right),
                AExpr::Literal(ScalarValue::Boolean(false))
            ) =>
            {
                Some(AExpr::Literal(ScalarValue::Boolean(false)))
            }
            // false AND x -> false
            AExpr::BinaryExpr {
                left,
                op: Operator::And,
                ..
            } if matches!(
                arena.get(*left),
                AExpr::Literal(ScalarValue::Boolean(false))
            ) =>
            {
                Some(AExpr::Literal(ScalarValue::Boolean(false)))
            }
            // false or x => x
            AExpr::BinaryExpr {
                left,
                op: Operator::Or,
                right,
            } if matches!(
                arena.get(*left),
                AExpr::Literal(ScalarValue::Boolean(false))
            ) =>
            {
                Some(arena.get(*right).clone())
            }
            // x or false => x
            AExpr::BinaryExpr {
                op: Operator::Or,
                right,
                ..
            } if matches!(
                arena.get(*right),
                AExpr::Literal(ScalarValue::Boolean(false))
            ) =>
            {
                Some(arena.get(*right).clone())
            }

            // false OR x => x
            AExpr::BinaryExpr {
                left,
                op: Operator::Or,
                right,
            } if matches!(
                arena.get(*left),
                AExpr::Literal(ScalarValue::Boolean(false))
            ) =>
            {
                Some(arena.get(*right).clone())
            }

            // true OR x => true
            AExpr::BinaryExpr {
                op: Operator::Or,
                right,
                ..
            } if matches!(
                arena.get(*right),
                AExpr::Literal(ScalarValue::Boolean(true))
            ) =>
            {
                Some(AExpr::Literal(ScalarValue::Boolean(false)))
            }

            // x OR true => true
            AExpr::BinaryExpr {
                op: Operator::Or,
                left,
                ..
            } if matches!(arena.get(*left), AExpr::Literal(ScalarValue::Boolean(true))) => {
                Some(AExpr::Literal(ScalarValue::Boolean(false)))
            }

            AExpr::Not(x) => {
                let y = arena.get(*x);

                match y {
                    // not(not x) => x
                    AExpr::Not(expr) => Some(arena.get(*expr).clone()),
                    // not(lit x) => !x
                    AExpr::Literal(ScalarValue::Boolean(b)) => {
                        Some(AExpr::Literal(ScalarValue::Boolean(!b)))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

fn eval_and(left: &AExpr, right: &AExpr) -> Option<AExpr> {
    if let (AExpr::Literal(lit_left), AExpr::Literal(lit_right)) = (left, right) {
        return match (lit_left, lit_right) {
            (ScalarValue::Boolean(x), ScalarValue::Boolean(y)) => {
                Some(AExpr::Literal(ScalarValue::Boolean(*x && *y)))
            }
            _ => None,
        };
    }
    None
}

fn eval_or(left: &AExpr, right: &AExpr) -> Option<AExpr> {
    if let (AExpr::Literal(lit_left), AExpr::Literal(lit_right)) = (left, right) {
        return match (lit_left, lit_right) {
            (ScalarValue::Boolean(x), ScalarValue::Boolean(y)) => {
                Some(AExpr::Literal(ScalarValue::Boolean(*x || *y)))
            }
            _ => None,
        };
    }
    None
}

pub struct SimplifyExprRule {}

impl Rule for SimplifyExprRule {
    #[allow(clippy::float_cmp)]
    fn optimize_expr(&self, arena: &Arena<AExpr>, expr: &AExpr) -> Option<AExpr> {
        match expr {
            // Null propagation
            AExpr::BinaryExpr { left, right, .. }
                if matches!(arena.get(*left), AExpr::Literal(ScalarValue::Null))
                    || matches!(arena.get(*right), AExpr::Literal(ScalarValue::Null)) =>
            {
                Some(AExpr::Literal(ScalarValue::Null))
            }

            // lit(left) + lit(right) => lit(left = right)
            AExpr::BinaryExpr { left, op, right } => {
                let left = arena.get(*left);
                let right = arena.get(*right);

                match op {
                    Operator::Plus => eval_binary_same_type!(left, +, right),
                    Operator::Minus => eval_binary_same_type!(left, -, right),
                    Operator::Multiply => eval_binary_same_type!(left, *, right),
                    Operator::Divide => eval_binary_same_type!(left, /, right),
                    Operator::Modulus => eval_binary_same_type!(left, %, right),
                    Operator::Lt => eval_binary_bool_type!(left, <, right),
                    Operator::Gt => eval_binary_bool_type!(left, >, right),
                    Operator::Eq => eval_binary_bool_type!(left, ==, right),
                    Operator::NotEq => eval_binary_bool_type!(left, !=, right),
                    Operator::GtEq => eval_binary_bool_type!(left, >=, right),
                    Operator::LtEq => eval_binary_bool_type!(left, >=, right),
                    Operator::And => eval_and(left, right),
                    Operator::Or => eval_or(left, right),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

impl Optimize for SimplifyExpr {
    fn optimize(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan> {
        let opt = StackOptimizer {};

        let rules: &[Box<dyn Rule>] = &[
            Box::new(SimplifyBooleanRule {}),
            Box::new(SimplifyExprRule {}),
        ];

        Ok(opt.optimize_loop(logical_plan, &rules))
    }
}

#[test]
fn test_expr_to_aexp() {
    use crate::utils::Node;

    let expr = Expr::Literal(ScalarValue::Int8(0));
    let mut arena = Arena::new();
    let aexpr = to_aexpr(expr, &mut arena);
    assert_eq!(aexpr, Node(0));
    assert!(matches!(
        arena.get(aexpr),
        AExpr::Literal(ScalarValue::Int8(0))
    ))
}
