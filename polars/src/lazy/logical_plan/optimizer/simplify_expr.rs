use crate::lazy::prelude::*;
use crate::prelude::*;

pub struct SimplifyExpr {}

// Evaluates x + y if possible
fn eval_plus(left: &Expr, right: &Expr) -> Option<Expr> {
    match (left, right) {
        (Expr::Literal(ScalarValue::Float32(x)), Expr::Literal(ScalarValue::Float32(y))) => {
            Some(Expr::Literal(ScalarValue::Float32(x + y)))
        }
        (Expr::Literal(ScalarValue::Float64(x)), Expr::Literal(ScalarValue::Float64(y))) => {
            Some(Expr::Literal(ScalarValue::Float64(x + y)))
        }
        (Expr::Literal(ScalarValue::Int8(x)), Expr::Literal(ScalarValue::Int8(y))) => {
            Some(Expr::Literal(ScalarValue::Int8(x + y)))
        }
        (Expr::Literal(ScalarValue::Int16(x)), Expr::Literal(ScalarValue::Int16(y))) => {
            Some(Expr::Literal(ScalarValue::Int16(x + y)))
        }
        (Expr::Literal(ScalarValue::Int32(x)), Expr::Literal(ScalarValue::Int32(y))) => {
            Some(Expr::Literal(ScalarValue::Int32(x + y)))
        }
        (Expr::Literal(ScalarValue::Int64(x)), Expr::Literal(ScalarValue::Int64(y))) => {
            Some(Expr::Literal(ScalarValue::Int64(x + y)))
        }
        (Expr::Literal(ScalarValue::UInt8(x)), Expr::Literal(ScalarValue::UInt8(y))) => {
            Some(Expr::Literal(ScalarValue::UInt8(x + y)))
        }
        (Expr::Literal(ScalarValue::UInt16(x)), Expr::Literal(ScalarValue::UInt16(y))) => {
            Some(Expr::Literal(ScalarValue::UInt16(x + y)))
        }
        (Expr::Literal(ScalarValue::UInt32(x)), Expr::Literal(ScalarValue::UInt32(y))) => {
            Some(Expr::Literal(ScalarValue::UInt32(x + y)))
        }
        (Expr::Literal(ScalarValue::UInt64(x)), Expr::Literal(ScalarValue::UInt64(y))) => {
            Some(Expr::Literal(ScalarValue::UInt64(x + y)))
        }

        _ => None,
    }
}

pub trait Rule {
    fn optimize_plan(&self, _logical_plan: &LogicalPlan) -> Option<LogicalPlan> {
        None
    }
    fn optimize_expr(&self, _expr: &Expr) -> Option<Expr> {
        None
    }
}

pub struct SimplifyExprRule {}

impl Rule for SimplifyExprRule {
    fn optimize_plan(&self, _logical_plan: &LogicalPlan) -> Option<LogicalPlan> {
        None
    }
    fn optimize_expr(&self, expr: &Expr) -> Option<Expr> {
        match expr {
            Expr::BinaryExpr {
                left,
                op: Operator::Plus,
                right,
            } => eval_plus(&left, &right),
            _ => None,
        }
    }
}

pub struct SimplifyBooleanRule {}

impl Rule for SimplifyBooleanRule {
    fn optimize_plan(&self, _logical_plan: &LogicalPlan) -> Option<LogicalPlan> {
        None
    }
    fn optimize_expr(&self, expr: &Expr) -> Option<Expr> {
        match expr {
            Expr::BinaryExpr {
                left,
                op: Operator::And,
                right,
            } if matches!(**left, Expr::Literal(ScalarValue::Boolean(true))) => {
                Some(*right.clone())
            }
            Expr::BinaryExpr {
                left,
                op: Operator::And,
                right,
            } if matches!(**right, Expr::Literal(ScalarValue::Boolean(true))) => {
                Some(*left.clone())
            }
            Expr::BinaryExpr {
                left,
                op: Operator::Or,
                right,
            } if matches!(**left, Expr::Literal(ScalarValue::Boolean(false))) => {
                Some(*right.clone())
            }
            Expr::BinaryExpr {
                left,
                op: Operator::Or,
                right,
            } if matches!(**right, Expr::Literal(ScalarValue::Boolean(false))) => {
                Some(*left.clone())
            }

            _ => None,
        }
    }
}

pub struct SimplifyOptimizer {}

impl SimplifyOptimizer {
    fn optimize_loop(&self, logical_plan: &mut LogicalPlan) {
        let rules: &[Box<dyn Rule>] = &[
            Box::new(SimplifyExprRule {}),
            Box::new(SimplifyBooleanRule {}),
        ];
        use Expr::*;
        use LogicalPlan::*;

        let mut changed = true;

        // run loop until reaching fixed point
        while changed {
            // recurse into sub plans and expressions and apply rules
            changed = false;
            let mut plans = vec![&mut *logical_plan];
            let mut exprs = vec![];

            while let Some(plan) = plans.pop() {
                // apply rules
                for rule in rules.iter() {
                    if let Some(x) = rule.optimize_plan(plan) {
                        *plan = x;
                        changed = true;
                    }
                }

                match plan {
                    Selection { input, predicate } => {
                        plans.push(input);
                        exprs.push(predicate);
                    }
                    Projection { expr, input, .. } => {
                        plans.push(input);
                        exprs.extend(expr);
                    }
                    DataFrameOp { input, .. } => {
                        plans.push(input);
                    }
                    Aggregate { input, aggs, .. } => {
                        plans.push(input);
                        exprs.extend(aggs);
                    }
                    Join {
                        input_left,
                        input_right,
                        ..
                    } => {
                        plans.push(input_left);
                        plans.push(input_right);
                    }
                    HStack {
                        input, exprs: e2, ..
                    } => {
                        plans.push(input);
                        exprs.extend(e2);
                    }
                    CsvScan { .. } | DataFrameScan { .. } => {}
                }

                while let Some(expr) = exprs.pop() {
                    for rule in rules.iter() {
                        if let Some(x) = rule.optimize_expr(expr) {
                            *expr = x;
                            changed = true;
                        }
                    }

                    match expr {
                        BinaryExpr { left, right, .. } => {
                            exprs.push(&mut *left);
                            exprs.push(&mut *right);
                        }
                        Alias(expr, ..) => {
                            exprs.push(expr);
                        }
                        Not(expr) => {
                            exprs.push(expr);
                        }
                        IsNotNull(expr) => {
                            exprs.push(expr);
                        }
                        IsNull(expr) => {
                            exprs.push(expr);
                        }
                        Cast { expr, .. } => {
                            exprs.push(expr);
                        }
                        Sort { expr, .. } => {
                            exprs.push(expr);
                        }
                        AggMin(expr) => {
                            exprs.push(expr);
                        }
                        AggMax(expr) => {
                            exprs.push(expr);
                        }
                        AggMedian(expr) => {
                            exprs.push(expr);
                        }
                        AggNUnique(expr) => {
                            exprs.push(expr);
                        }
                        AggFirst(expr) => {
                            exprs.push(expr);
                        }
                        AggLast(expr) => {
                            exprs.push(expr);
                        }
                        AggList(expr) => {
                            exprs.push(expr);
                        }
                        AggMean(expr) => {
                            exprs.push(expr);
                        }
                        AggQuantile { expr, .. } => {
                            exprs.push(expr);
                        }
                        AggSum(expr) => {
                            exprs.push(expr);
                        }
                        AggGroups(expr) => {
                            exprs.push(expr);
                        }
                        AggCount(expr) => {
                            exprs.push(expr);
                        }
                        Shift { input, .. } => {
                            exprs.push(input);
                        }
                        Ternary {
                            predicate,
                            truthy,
                            falsy,
                        } => {
                            exprs.push(predicate);
                            exprs.push(truthy);
                            exprs.push(falsy);
                        }
                        Apply { input, .. } => {
                            exprs.push(input);
                        }
                        Literal { .. } | Column { .. } | Wildcard => {}
                    }
                }
            }
        }
    }
}

impl Optimize for SimplifyExpr {
    fn optimize(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan> {
        let opt = SimplifyOptimizer {};
        let mut plan = logical_plan;
        opt.optimize_loop(&mut plan);
        Ok(plan)
    }
}
