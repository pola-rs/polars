use crate::lazy::logical_plan::*;
use crate::lazy::prelude::*;
use crate::prelude::*;
use std::sync::Arc;

struct Arena<T> {
    items: Vec<T>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Node(usize);

// Simple Arena
// Allocates memory and stores item in a Vec. Only deallocates when being dropped itself.
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
// AExpr representation of Nodes which are allocated in an Arena
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

// ALogicalPlan is a representation of LogicalPlan with Nodes which are allocated in an Arena
enum ALogicalPlan {
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

fn to_alp(lp: LogicalPlan, _arena: &mut Arena<AExpr>, lp_arena: &mut Arena<ALogicalPlan>) -> Node {
    let v = match lp {
        LogicalPlan::Selection { input, predicate } => {
            let i = to_alp(*input, _arena, lp_arena);
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
            let i = to_alp(*input, _arena, lp_arena);
            ALogicalPlan::Projection {
                expr: exp,
                input: i,
                schema: schema.clone(),
            }
        }
        LogicalPlan::DataFrameOp { input, operation } => {
            let i = to_alp(*input, _arena, lp_arena);
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
            let i = to_alp(*input, _arena, lp_arena);
            let aggs_new = aggs.iter().map(|x| to_aexpr(x.clone(), _arena)).collect();

            ALogicalPlan::Aggregate {
                input: i,
                keys: keys.clone(),
                aggs: aggs_new,
                schema: schema.clone(),
            }
        }
        LogicalPlan::Join {
            input_left,
            input_right,
            schema,
            how,
            left_on,
            right_on,
        } => {
            let i_l = to_alp(*input_left, _arena, lp_arena);
            let i_r = to_alp(*input_right, _arena, lp_arena);

            let l_on = to_aexpr(left_on, _arena);

            let r_on = to_aexpr(right_on, _arena);

            ALogicalPlan::Join {
                input_left: i_l,
                input_right: i_r,
                schema: schema.clone(),
                left_on: l_on,
                how,
                right_on: r_on,
            }
        }
        LogicalPlan::HStack {
            input,
            exprs,
            schema,
        } => {
            let exp = exprs.iter().map(|x| to_aexpr(x.clone(), _arena)).collect();
            let i = to_alp(*input, _arena, lp_arena);
            ALogicalPlan::HStack {
                input: i,
                exprs: exp,
                schema: schema.clone(),
            }
        }
    };
    lp_arena.add(v)
}

fn node_to_exp(node: Node, _arena: &Arena<AExpr>) -> Expr {
    let expr = _arena.get(node);

    match expr {
        AExpr::Alias(expr, name) => {
            let exp = node_to_exp(*expr, _arena);
            Expr::Alias(Box::new(exp), name.clone())
        }
        AExpr::Column(a) => Expr::Column(a.clone()),
        AExpr::Literal(s) => Expr::Literal(s.clone()),
        AExpr::BinaryExpr { left, op, right } => {
            let l = node_to_exp(*left, _arena);
            let r = node_to_exp(*right, _arena);
            Expr::BinaryExpr {
                left: Box::new(l),
                op: *op,
                right: Box::new(r),
            }
        }
        AExpr::Not(expr) => {
            let exp = node_to_exp(*expr, _arena);
            Expr::Not(Box::new(exp))
        }
        AExpr::IsNotNull(expr) => {
            let exp = node_to_exp(*expr, _arena);
            Expr::IsNotNull(Box::new(exp))
        }
        AExpr::IsNull(expr) => {
            let exp = node_to_exp(*expr, _arena);
            Expr::IsNull(Box::new(exp))
        }
        AExpr::Cast { expr, data_type } => {
            let exp = node_to_exp(*expr, _arena);
            Expr::Cast {
                expr: Box::new(exp),
                data_type: data_type.clone(),
            }
        }
        AExpr::Sort { expr, reverse } => {
            let exp = node_to_exp(*expr, _arena);
            Expr::Sort {
                expr: Box::new(exp),
                reverse: *reverse,
            }
        }
        AExpr::AggMin(expr) => {
            let exp = node_to_exp(*expr, _arena);
            Expr::AggMin(Box::new(exp))
        }
        AExpr::AggMax(expr) => {
            let exp = node_to_exp(*expr, _arena);
            Expr::AggMax(Box::new(exp))
        }

        AExpr::AggMedian(expr) => {
            let exp = node_to_exp(*expr, _arena);
            Expr::AggMedian(Box::new(exp))
        }
        AExpr::AggNUnique(expr) => {
            let exp = node_to_exp(*expr, _arena);
            Expr::AggNUnique(Box::new(exp))
        }
        AExpr::AggFirst(expr) => {
            let exp = node_to_exp(*expr, _arena);
            Expr::AggFirst(Box::new(exp))
        }
        AExpr::AggLast(expr) => {
            let exp = node_to_exp(*expr, _arena);
            Expr::AggLast(Box::new(exp))
        }
        AExpr::AggMean(expr) => {
            let exp = node_to_exp(*expr, _arena);
            Expr::AggMean(Box::new(exp))
        }
        AExpr::AggList(expr) => {
            let exp = node_to_exp(*expr, _arena);
            Expr::AggList(Box::new(exp))
        }
        AExpr::AggQuantile { expr, quantile } => {
            let exp = node_to_exp(*expr, _arena);
            Expr::AggQuantile {
                expr: Box::new(exp),
                quantile: *quantile,
            }
        }
        AExpr::AggSum(expr) => {
            let exp = node_to_exp(*expr, _arena);
            Expr::AggSum(Box::new(exp))
        }
        AExpr::AggGroups(expr) => {
            let exp = node_to_exp(*expr, _arena);
            Expr::AggGroups(Box::new(exp))
        }
        AExpr::Shift { input, periods } => {
            let e = node_to_exp(*input, _arena);
            Expr::Shift {
                input: Box::new(e),
                periods: *periods,
            }
        }
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            let p = node_to_exp(*predicate, _arena);
            let t = node_to_exp(*truthy, _arena);
            let f = node_to_exp(*falsy, _arena);

            Expr::Ternary {
                predicate: Box::new(p),
                truthy: Box::new(t),
                falsy: Box::new(f),
            }
        }
        AExpr::Apply {
            input,
            function,
            output_type,
        } => {
            let i = node_to_exp(*input, _arena);
            Expr::Apply {
                input: Box::new(i),
                function: function.clone(),
                output_type: output_type.clone(),
            }
        }

        AExpr::Wildcard => Expr::Wildcard,
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
        ALogicalPlan::DataFrameOp { input, operation } => {
            let lp = node_to_lp(*input, _arena, lp_arena);

            LogicalPlan::DataFrameOp {
                input: Box::new(lp),
                operation: operation.clone(),
            }
        }
        ALogicalPlan::Aggregate {
            input,
            keys,
            aggs,
            schema,
        } => {
            let i = node_to_lp(*input, _arena, lp_arena);
            let a = aggs.iter().map(|x| node_to_exp(*x, _arena)).collect();

            LogicalPlan::Aggregate {
                input: Box::new(i),
                keys: keys.clone(),
                aggs: a,
                schema: schema.clone(),
            }
        }
        ALogicalPlan::Join {
            input_left,
            input_right,
            schema,
            how,
            left_on,
            right_on,
        } => {
            let i_l = node_to_lp(*input_left, _arena, lp_arena);
            let i_r = node_to_lp(*input_right, _arena, lp_arena);

            let l_on = node_to_exp(*left_on, _arena);
            let r_on = node_to_exp(*right_on, _arena);

            LogicalPlan::Join {
                input_left: Box::new(i_l),
                input_right: Box::new(i_r),
                schema: schema.clone(),
                how: how.clone(),
                left_on: l_on,
                right_on: r_on,
            }
        }
        ALogicalPlan::HStack {
            input,
            exprs,
            schema,
        } => {
            let i = node_to_lp(*input, _arena, lp_arena);
            let e = exprs.iter().map(|x| node_to_exp(*x, _arena)).collect();

            LogicalPlan::HStack {
                input: Box::new(i),
                exprs: e,
                schema: schema.clone(),
            }
        }
        _ => unimplemented!(""),
    }
}

// Evaluates x + y if possible
fn eval_plus(left: &AExpr, right: &AExpr) -> Option<AExpr> {
    if let (AExpr::Literal(lit_left), AExpr::Literal(lit_right)) = (left, right) {
        return match (lit_left, lit_right) {
            (ScalarValue::Float32(x), ScalarValue::Float32(y)) => {
                Some(AExpr::Literal(ScalarValue::Float32(x + y)))
            }
            (ScalarValue::Float64(x), ScalarValue::Float64(y)) => {
                Some(AExpr::Literal(ScalarValue::Float64(x + y)))
            }
            (ScalarValue::Int8(x), ScalarValue::Int8(y)) => {
                Some(AExpr::Literal(ScalarValue::Int8(x + y)))
            }
            (ScalarValue::Int16(x), ScalarValue::Int16(y)) => {
                Some(AExpr::Literal(ScalarValue::Int16(x + y)))
            }
            (ScalarValue::Int32(x), ScalarValue::Int32(y)) => {
                Some(AExpr::Literal(ScalarValue::Int32(x + y)))
            }
            (ScalarValue::Int64(x), ScalarValue::Int64(y)) => {
                Some(AExpr::Literal(ScalarValue::Int64(x + y)))
            }
            (ScalarValue::UInt8(x), ScalarValue::UInt8(y)) => {
                Some(AExpr::Literal(ScalarValue::UInt8(x + y)))
            }
            (ScalarValue::UInt16(x), ScalarValue::UInt16(y)) => {
                Some(AExpr::Literal(ScalarValue::UInt16(x + y)))
            }
            (ScalarValue::UInt32(x), ScalarValue::UInt32(y)) => {
                Some(AExpr::Literal(ScalarValue::UInt32(x + y)))
            }
            (ScalarValue::UInt64(x), ScalarValue::UInt64(y)) => {
                Some(AExpr::Literal(ScalarValue::UInt64(x + y)))
            }
            _ => None,
        };
    }
    None
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

            _ => None,
        }
    }
}

pub struct SimplifyExprRule {}

impl Rule for SimplifyExprRule {
    fn optimize_expr(&self, arena: &Arena<AExpr>, expr: &AExpr) -> Option<AExpr> {
        match expr {
            // Null propagation
            AExpr::BinaryExpr {
                left,
                op: Operator::Plus,
                right,
            } if matches!(arena.get(*left), AExpr::Literal(ScalarValue::Null))
                || matches!(arena.get(*right), AExpr::Literal(ScalarValue::Null)) =>
            {
                Some(AExpr::Literal(ScalarValue::Null))
            }

            AExpr::BinaryExpr {
                left,
                op: Operator::Plus,
                right,
            } => {
                let left = arena.get(*left);
                let right = arena.get(*right);

                eval_plus(left, right)
            }
            _ => None,
        }
    }
}

pub struct SimplifyOptimizer {}

impl SimplifyOptimizer {
    fn optimize_loop(&self, logical_plan: LogicalPlan) -> LogicalPlan {
        let rules: &[Box<dyn Rule>] = &[
            Box::new(SimplifyBooleanRule {}),
            Box::new(SimplifyExprRule {}),
        ];

        let mut changed = true;

        // initialize arena
        let mut expr_arena = Arena::new();
        let mut lp_arena = Arena::new();
        let lp_top = to_alp(logical_plan, &mut expr_arena, &mut lp_arena);

        let mut plans = Vec::new();
        let mut exprs = Vec::new();

        // run loop until reaching fixed point
        while changed {
            // recurse into sub plans and expressions and apply rules
            changed = false;
            plans.push(lp_top);
            while let Some(node) = plans.pop() {
                // apply rules
                for rule in rules.iter() {
                    // keep iterating over same rule
                    while let Some(x) = rule.optimize_plan(&lp_arena, &lp_arena.get(node)) {
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
                        // keep iterating over same rule
                        while let Some(x) = rule.optimize_expr(&expr_arena, &expr_arena.get(node)) {
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
