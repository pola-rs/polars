use crate::lazy::logical_plan::*;
use crate::lazy::prelude::*;
use crate::prelude::*;
use std::sync::Arc;
pub struct SimplifyExpr {}

struct Arena<T> {
    items: Vec<T>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Node(usize);

impl<T> Arena<T> {
    fn add(&mut self, val: T) -> Node {
        let idx = self.items.len();
        self.items.push(val);
        Node(idx)
    }

    pub fn new() -> Self {
        Arena { items: vec![] }
    }

    pub fn get(&mut self, idx: Node) -> &T {
        unsafe { self.items.get_unchecked(idx.0) }
    }

    pub fn get_mut(&mut self, idx: Node) -> &mut T {
        unsafe { self.items.get_unchecked_mut(idx.0) }
    }

    pub fn assign(&mut self, idx: Node, val: T) {
        let x = self.get_mut(idx);
        *x = val;
    }
}
// Store pointers to values in an arena
#[derive(Clone)]
enum AExpr {
    Alias(Node, Arc<String>),
    Column(Arc<String>),
    Literal(ScalarValue),
    BinaryExpr {
        left: Node,
        op: Operator,
        right: Node,
    },
    Not(Node),
    IsNotNull(Node),
    IsNull(Node),
    Cast {
        expr: Node,
        data_type: ArrowDataType,
    },
    Sort {
        expr: Node,
        reverse: bool,
    },
    AggMin(Node),
    AggMax(Node),
    AggMedian(Node),
    AggNUnique(Node),
    AggFirst(Node),
    AggLast(Node),
    AggMean(Node),
    AggList(Node),
    AggQuantile {
        expr: Node,
        quantile: f64,
    },
    AggSum(Node),
    AggGroups(Node),
    Ternary {
        predicate: Node,
        truthy: Node,
        falsy: Node,
    },
    Apply {
        input: Node,
        function: Arc<dyn Udf>,
        output_type: Option<ArrowDataType>,
    },
    Shift {
        input: Node,
        periods: Node,
    },
    Wildcard,
}

enum ALogicalPlan {
    // filter on a boolean mask
    Selection {
        input: Node,
        predicate: AExpr,
    },
    CsvScan {
        path: String,
        schema: Schema,
        has_header: bool,
        delimiter: Option<u8>,
    },
    DataFrameScan {
        df: Arc<Mutex<DataFrame>>,
        schema: Schema,
    },
    // vertical selection
    Projection {
        expr: Vec<AExpr>,
        input: Node,
        schema: Schema,
    },
    DataFrameOp {
        input: Node,
        operation: DataFrameOperation,
    },
    Aggregate {
        input: Node,
        keys: Arc<Vec<String>>,
        aggs: Vec<AExpr>,
        schema: Schema,
    },
    Join {
        input_left: Node,
        input_right: Node,
        schema: Schema,
        how: JoinType,
        left_on: AExpr,
        right_on: AExpr,
    },
    HStack {
        input: Node,
        exprs: Vec<AExpr>,
        schema: Schema,
    },
}
// converts expression to AExpr, which uses an arena (Vec) for allocation
fn to_aexpr(expr: Expr, arena: &mut Arena<AExpr>) -> Node {
    match expr {
        Expr::Alias(e, name) => {
            let v = AExpr::Alias(to_aexpr(*e, arena), name);
            arena.add(v)
        }
        Expr::Literal(value) => {
            let v = AExpr::Literal(value);
            arena.add(v)
        }
        Expr::Column(s) => {
            let v = AExpr::Column(s.clone());
            arena.add(v)
        }
        Expr::BinaryExpr { left, op, right } => {
            let l = to_aexpr(*left, arena);
            let r = to_aexpr(*right, arena);
            let v = AExpr::BinaryExpr {
                left: l,
                op: op,
                right: r,
            };

            arena.add(v)
        }
        Expr::Not(e) => {
            let v = AExpr::Not(to_aexpr(*e, arena));
            arena.add(v)
        }
        Expr::IsNotNull(e) => {
            let v = AExpr::IsNotNull(to_aexpr(*e, arena));
            arena.add(v)
        }
        Expr::IsNull(e) => {
            let v = AExpr::IsNull(to_aexpr(*e, arena));
            arena.add(v)
        }

        _ => unimplemented!("TODO"),
    }
}

fn lp_to_aexpr(
    lp: &LogicalPlan,
    arena: &mut Arena<AExpr>,
    lp_arena: &mut Arena<ALogicalPlan>,
) -> Node {
    match lp {
        _ => unimplemented!("TODO"),
    }
}

pub struct ExampleBooleanRule {}

impl ExampleBooleanRule {
    fn optimize_expr(&self, arena: &mut Arena<AExpr>, expr: AExpr) -> Option<AExpr> {
        match expr {
            AExpr::BinaryExpr {
                left,
                op: Operator::And,
                right,
            } if matches!(arena.get(left), AExpr::Literal(ScalarValue::Boolean(true))) => {
                Some(arena.get(right).clone())
            }
            _ => None,
        }
    }
}

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

        // initialize arena
        let mut expr_arena = Arena::new();
        let mut lp_arena = Arena::new();
        let lp_top = lp_to_aexpr(logical_plan, &mut expr_arena, &mut lp_arena);

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

#[test]
fn test_expr_to_aexp() {
    let expr = Expr::Literal(ScalarValue::Int8(0));
    let mut arena = Arena::new();
    let aexpr = to_aexpr(expr, &mut arena);
    assert_eq!(aexpr, Node(0));
    assert!(matches!(
        arena.get(aexpr),
        AExpr::Literal(ScalarValue::Int8(0))
    ))
}
