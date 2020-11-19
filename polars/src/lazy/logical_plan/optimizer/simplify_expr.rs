use crate::lazy::logical_plan::*;
use crate::lazy::prelude::*;
use crate::prelude::*;
use std::sync::Arc;

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

    fn new() -> Self {
        Arena { items: vec![] }
    }

    fn get(&self, idx: Node) -> &T {
        unsafe { self.items.get_unchecked(idx.0) }
    }

    fn get_mut(&mut self, idx: Node) -> &mut T {
        unsafe { self.items.get_unchecked_mut(idx.0) }
    }

    fn assign(&mut self, idx: Node, val: T) {
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
        periods: i32,
    },
    Wildcard,
}

enum ALogicalPlan {
    // filter on a boolean mask
    Selection {
        input: Node,
        predicate: Node,
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
    Projection {
        expr: Vec<Node>,
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
        aggs: Vec<Node>,
        schema: Schema,
    },
    Join {
        input_left: Node,
        input_right: Node,
        schema: Schema,
        how: JoinType,
        left_on: Node,
        right_on: Node,
    },
    HStack {
        input: Node,
        exprs: Vec<Node>,
        schema: Schema,
    },
}
// converts expression to AExpr, which uses an arena (Vec) for allocation
fn to_aexpr(expr: Expr, arena: &mut Arena<AExpr>) -> Node {
    let v = match expr {
        Expr::Alias(e, name) => AExpr::Alias(to_aexpr(*e, arena), name),
        Expr::Literal(value) => AExpr::Literal(value),
        Expr::Column(s) => AExpr::Column(s),
        Expr::BinaryExpr { left, op, right } => {
            let l = to_aexpr(*left, arena);
            let r = to_aexpr(*right, arena);
            AExpr::BinaryExpr {
                left: l,
                op,
                right: r,
            }
        }
        Expr::Not(e) => AExpr::Not(to_aexpr(*e, arena)),
        Expr::IsNotNull(e) => AExpr::IsNotNull(to_aexpr(*e, arena)),
        Expr::IsNull(e) => AExpr::IsNull(to_aexpr(*e, arena)),

        Expr::Cast { expr, data_type } => AExpr::Cast {
            expr: to_aexpr(*expr, arena),
            data_type,
        },
        Expr::Sort { expr, reverse } => AExpr::Sort {
            expr: to_aexpr(*expr, arena),
            reverse,
        },
        Expr::AggMin(expr) => AExpr::AggMin(to_aexpr(*expr, arena)),
        Expr::AggMax(expr) => AExpr::AggMax(to_aexpr(*expr, arena)),
        Expr::AggMedian(expr) => AExpr::AggMedian(to_aexpr(*expr, arena)),
        Expr::AggNUnique(expr) => AExpr::AggNUnique(to_aexpr(*expr, arena)),
        Expr::AggFirst(expr) => AExpr::AggFirst(to_aexpr(*expr, arena)),
        Expr::AggLast(expr) => AExpr::AggLast(to_aexpr(*expr, arena)),
        Expr::AggMean(expr) => AExpr::AggMean(to_aexpr(*expr, arena)),
        Expr::AggList(expr) => AExpr::AggList(to_aexpr(*expr, arena)),
        Expr::AggQuantile { expr, quantile } => AExpr::AggQuantile {
            expr: to_aexpr(*expr, arena),
            quantile,
        },
        Expr::AggSum(expr) => AExpr::AggSum(to_aexpr(*expr, arena)),
        Expr::AggGroups(expr) => AExpr::AggGroups(to_aexpr(*expr, arena)),
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            let p = to_aexpr(*predicate, arena);
            let t = to_aexpr(*truthy, arena);
            let f = to_aexpr(*falsy, arena);
            AExpr::Ternary {
                predicate: p,
                truthy: t,
                falsy: f,
            }
        }
        Expr::Apply {
            input,
            function,
            output_type,
        } => AExpr::Apply {
            input: to_aexpr(*input, arena),
            function,
            output_type,
        },
        Expr::Shift { input, periods } => AExpr::Shift {
            input: to_aexpr(*input, arena),
            periods,
        },
        Expr::Wildcard => AExpr::Wildcard,
    };
    arena.add(v)
}

fn lp_to_aexpr(
    lp: LogicalPlan,
    _arena: &mut Arena<AExpr>,
    lp_arena: &mut Arena<ALogicalPlan>,
) -> Node {
    let v = match lp {
        LogicalPlan::Selection { input, predicate } => {
            let i = lp_to_aexpr(*input, _arena, lp_arena);
            let p = to_aexpr(predicate, _arena);
            ALogicalPlan::Selection {
                input: i,
                predicate: p,
            }
        }
        CsvScan {
            path,
            schema,
            has_header,
            delimiter,
        } => ALogicalPlan::CsvScan {
            path: path.clone(),
            schema: schema.clone(),
            has_header: has_header,
            delimiter: delimiter,
        },
        LogicalPlan::DataFrameScan { df, schema } => ALogicalPlan::DataFrameScan {
            df: df.clone(),
            schema: schema.clone(),
        },
        LogicalPlan::Projection {
            expr,
            input,
            schema,
        } => {
            let exp = expr.iter().map(|x| to_aexpr(x.clone(), _arena)).collect();
            let i = lp_to_aexpr(*input, _arena, lp_arena);
            ALogicalPlan::Projection {
                expr: exp,
                input: i,
                schema: schema.clone(),
            }
        }
        LogicalPlan::DataFrameOp { input, operation } => {
            let i = lp_to_aexpr(*input, _arena, lp_arena);
            ALogicalPlan::DataFrameOp {
                input: i,
                operation: operation,
            }
        }
        LogicalPlan::Aggregate {
            input,
            keys,
            aggs,
            schema,
        } => {
            let i = lp_to_aexpr(*input, _arena, lp_arena);
            let aggs_new = aggs.iter().map(|x| to_aexpr(x.clone(), _arena)).collect();

            ALogicalPlan::Aggregate {
                input: i,
                keys: keys.clone(),
                aggs: aggs_new,
                schema: schema.clone(),
            }
        }
        // LogicalPlan::Join {
        //     input_left,
        //     input_right,
        //     schema,
        //     how,
        //     left_on,
        //     right_on,
        // } => {}
        // LogicalPlan::HStack {
        //     input,
        //     exprs,
        //     schema,
        // } => {}
        _ => unimplemented!("TODO"),
    };
    lp_arena.add(v)
}

fn node_to_exp(node: Node, _arena: &Arena<AExpr>) -> Expr {
    let expr = _arena.get(node);

    match expr {
        // AExpr::Alias(_, _) => {}
        // AExpr::Column(_) => {}
        // AExpr::Literal(_) => {}
        // AExpr::BinaryExpr { left, op, right } => {}
        // AExpr::Not(_) => {}
        // AExpr::IsNotNull(_) => {}
        // AExpr::IsNull(_) => {}
        // AExpr::Cast { expr, data_type } => {}
        // AExpr::Sort { expr, reverse } => {}
        // AExpr::AggMin(_) => {}
        // AExpr::AggMax(_) => {}
        // AExpr::AggMedian(_) => {}
        // AExpr::AggNUnique(_) => {}
        // AExpr::AggFirst(_) => {}
        // AExpr::AggLast(_) => {}
        // AExpr::AggMean(_) => {}
        // AExpr::AggList(_) => {}
        // AExpr::AggQuantile { expr, quantile } => {}
        // AExpr::AggSum(_) => {}
        // AExpr::AggGroups(_) => {}
        // AExpr::Ternary {
        //     predicate,
        //     truthy,
        //     falsy,
        // } => {}
        // AExpr::Apply {
        //     input,
        //     function,
        //     output_type,
        // } => {}
        // AExpr::Shift { input, periods } => {}
        AExpr::Wildcard => Expr::Wildcard,
        _ => unimplemented!(""),
    }
}

fn node_to_lp(node: Node, _arena: &Arena<AExpr>, lp_arena: &Arena<ALogicalPlan>) -> LogicalPlan {
    let lp = lp_arena.get(node);

    match lp {
        ALogicalPlan::Selection { input, predicate } => {
            let lp = node_to_lp(*input, _arena, lp_arena);
            let p = node_to_exp(*predicate, _arena);
            LogicalPlan::Selection {
                input: Box::new(lp),
                predicate: p,
            }
        }
        // ALogicalPlan::CsvScan {
        //     path,
        //     schema,
        //     has_header,
        //     delimiter,
        // } => {}
        ALogicalPlan::DataFrameScan { df, schema } => LogicalPlan::DataFrameScan {
            df: df.clone(),
            schema: schema.clone(),
        },
        ALogicalPlan::Projection {
            expr,
            input,
            schema,
        } => {
            let exprs = expr.iter().map(|x| node_to_exp(*x, _arena)).collect();
            let i = node_to_lp(*input, _arena, lp_arena);

            LogicalPlan::Projection {
                expr: exprs,
                input: Box::new(i),
                schema: schema.clone(),
            }
        }
        // ALogicalPlan::DataFrameOp { input, operation } => {}
        // ALogicalPlan::Aggregate {
        //     input,
        //     keys,
        //     aggs,
        //     schema,
        // } => {}
        // ALogicalPlan::Join {
        //     input_left,
        //     input_right,
        //     schema,
        //     how,
        //     left_on,
        //     right_on,
        // } => {}
        // ALogicalPlan::HStack {
        //     input,
        //     exprs,
        //     schema,
        // } => {}
        _ => unimplemented!(""),
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
pub struct SimplifyExpr {}

trait Rule {
    fn optimize_plan(
        &self,
        _arena: &Arena<ALogicalPlan>,
        _logical_plan: &ALogicalPlan,
    ) -> Option<ALogicalPlan> {
        None
    }
    fn optimize_expr(&self, _arena: &Arena<AExpr>, _expr: &AExpr) -> Option<AExpr> {
        None
    }
}

struct SimplifyBooleanRule {}

impl Rule for SimplifyBooleanRule {
    fn optimize_expr(&self, arena: &Arena<AExpr>, expr: &AExpr) -> Option<AExpr> {
        match expr {
            AExpr::BinaryExpr {
                left,
                op: Operator::And,
                right,
            } if matches!(arena.get(*left), AExpr::Literal(ScalarValue::Boolean(true))) => {
                Some(arena.get(*right).clone())
            }
            _ => None,
        }
    }
}

// pub struct SimplifyExprRule {}

// impl Rule for SimplifyExprRule {
//     fn optimize_plan(&self, _logical_plan: &LogicalPlan) -> Option<LogicalPlan> {
//         None
//     }
//     fn optimize_expr(&self, expr: &Expr) -> Option<Expr> {
//         match expr {
//             Expr::BinaryExpr {
//                 left,
//                 op: Operator::Plus,
//                 right,
//             } => eval_plus(&left, &right),
//             _ => None,
//         }
//     }
// }

// pub struct SimplifyBooleanRule {}

// impl Rule for SimplifyBooleanRule {
//     fn optimize_plan(&self, _logical_plan: &LogicalPlan) -> Option<LogicalPlan> {
//         None
//     }
//     fn optimize_expr(&self, expr: &Expr) -> Option<Expr> {
//         match expr {
//             Expr::BinaryExpr {
//                 left,
//                 op: Operator::And,
//                 right,
//             } if matches!(**left, Expr::Literal(ScalarValue::Boolean(true))) => {
//                 Some(*right.clone())
//             }
//             Expr::BinaryExpr {
//                 left,
//                 op: Operator::And,
//                 right,
//             } if matches!(**right, Expr::Literal(ScalarValue::Boolean(true))) => {
//                 Some(*left.clone())
//             }
//             Expr::BinaryExpr {
//                 left,
//                 op: Operator::Or,
//                 right,
//             } if matches!(**left, Expr::Literal(ScalarValue::Boolean(false))) => {
//                 Some(*right.clone())
//             }
//             Expr::BinaryExpr {
//                 left,
//                 op: Operator::Or,
//                 right,
//             } if matches!(**right, Expr::Literal(ScalarValue::Boolean(false))) => {
//                 Some(*left.clone())
//             }

//             _ => None,
//         }
//     }
// }

pub struct SimplifyOptimizer {}

impl SimplifyOptimizer {
    fn optimize_loop(&self, logical_plan: LogicalPlan) -> LogicalPlan {
        let rules: &[Box<dyn Rule>] = &[Box::new(SimplifyBooleanRule {})];

        let mut changed = true;

        // initialize arena
        let mut expr_arena = Arena::new();
        let mut lp_arena = Arena::new();
        let lp_top = lp_to_aexpr(logical_plan, &mut expr_arena, &mut lp_arena);

        let mut plans = Vec::with_capacity(lp_arena.items.len());
        let mut exprs = Vec::with_capacity(expr_arena.items.len());

        // run loop until reaching fixed point
        while changed {
            // recurse into sub plans and expressions and apply rules
            changed = false;
            plans.push(lp_top);
            while let Some(node) = plans.pop() {
                // apply rules
                for rule in rules.iter() {
                    if let Some(x) = rule.optimize_plan(&lp_arena, &lp_arena.get(node)) {
                        lp_arena.assign(node, x);
                        changed = true;
                    }
                }

                let plan = lp_arena.get(node);

                match plan {
                    ALogicalPlan::Selection { input, predicate } => {
                        plans.push(*input);
                        exprs.push(*predicate);
                    }
                    ALogicalPlan::Projection { expr, input, .. } => {
                        plans.push(*input);
                        exprs.extend(expr);
                    }
                    ALogicalPlan::DataFrameOp { input, .. } => {
                        plans.push(*input);
                    }
                    ALogicalPlan::Aggregate { input, aggs, .. } => {
                        plans.push(*input);
                        exprs.extend(aggs);
                    }
                    ALogicalPlan::Join {
                        input_left,
                        input_right,
                        ..
                    } => {
                        plans.push(*input_left);
                        plans.push(*input_right);
                    }
                    ALogicalPlan::HStack {
                        input, exprs: e2, ..
                    } => {
                        plans.push(*input);
                        exprs.extend(e2);
                    }
                    ALogicalPlan::CsvScan { .. } | ALogicalPlan::DataFrameScan { .. } => {}
                }

                while let Some(node) = exprs.pop() {
                    for rule in rules.iter() {
                        if let Some(x) = rule.optimize_expr(&expr_arena, &expr_arena.get(node)) {
                            expr_arena.assign(node, x);
                            changed = true;
                        }
                    }
                    let expr = expr_arena.get(node);

                    match expr {
                        AExpr::BinaryExpr { left, right, .. } => {
                            exprs.push(*left);
                            exprs.push(*right);
                        }
                        AExpr::Alias(expr, ..) => {
                            exprs.push(*expr);
                        }
                        AExpr::Not(expr) => {
                            exprs.push(*expr);
                        }
                        AExpr::IsNotNull(expr) => {
                            exprs.push(*expr);
                        }
                        AExpr::IsNull(expr) => {
                            exprs.push(*expr);
                        }
                        AExpr::Cast { expr, .. } => {
                            exprs.push(*expr);
                        }
                        AExpr::Sort { expr, .. } => {
                            exprs.push(*expr);
                        }
                        AExpr::AggMin(expr) => {
                            exprs.push(*expr);
                        }
                        AExpr::AggMax(expr) => {
                            exprs.push(*expr);
                        }
                        AExpr::AggMedian(expr) => {
                            exprs.push(*expr);
                        }
                        AExpr::AggNUnique(expr) => {
                            exprs.push(*expr);
                        }
                        AExpr::AggFirst(expr) => {
                            exprs.push(*expr);
                        }
                        AExpr::AggLast(expr) => {
                            exprs.push(*expr);
                        }
                        AExpr::AggList(expr) => {
                            exprs.push(*expr);
                        }
                        AExpr::AggMean(expr) => {
                            exprs.push(*expr);
                        }
                        AExpr::AggQuantile { expr, .. } => {
                            exprs.push(*expr);
                        }
                        AExpr::AggSum(expr) => {
                            exprs.push(*expr);
                        }
                        AExpr::AggGroups(expr) => {
                            exprs.push(*expr);
                        }
                        AExpr::Shift { input, .. } => {
                            exprs.push(*input);
                        }
                        AExpr::Ternary {
                            predicate,
                            truthy,
                            falsy,
                        } => {
                            exprs.push(*predicate);
                            exprs.push(*truthy);
                            exprs.push(*falsy);
                        }
                        AExpr::Apply { input, .. } => {
                            exprs.push(*input);
                        }
                        AExpr::Literal { .. } | AExpr::Column { .. } | AExpr::Wildcard => {}
                    }
                }
            }
        }

        node_to_lp(lp_top, &expr_arena, &lp_arena)
    }
}

impl Optimize for SimplifyExpr {
    fn optimize(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan> {
        let opt = SimplifyOptimizer {};
        Ok(opt.optimize_loop(logical_plan))
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
