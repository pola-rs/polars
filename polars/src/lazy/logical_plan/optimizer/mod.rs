use crate::lazy::logical_plan::Context;
use crate::lazy::prelude::*;
use crate::lazy::utils::expr_to_root_column_expr;
use crate::prelude::*;
use crate::utils::{get_supertype, Arena, Node};
use ahash::RandomState;
use arrow::datatypes::SchemaRef;
use std::collections::HashMap;

pub(crate) mod aggregate_scan_projections;
pub(crate) mod predicate;
pub(crate) mod projection;
pub(crate) mod simplify_expr;
pub(crate) mod type_coercion;

// check if a selection/projection can be done on the downwards schema
fn check_down_node(expr: &Expr, down_schema: &Schema) -> bool {
    match expr_to_root_column_expr(expr) {
        Err(_) => false,
        Ok(e) => e.to_field(down_schema, Context::Other).is_ok(),
    }
}

pub trait Optimize {
    fn optimize(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan>;
}

// arbitrary constant to reduce reallocation.
// don't expect more than 100 predicates.
const HASHMAP_SIZE: usize = 100;

fn init_hashmap<K, V>() -> HashMap<K, V, RandomState> {
    HashMap::with_capacity_and_hasher(HASHMAP_SIZE, RandomState::new())
}

/// Optimizer that uses a stack and memory arenas in favor of recursion
pub struct StackOptimizer {}

impl StackOptimizer {
    fn optimize_loop(&self, logical_plan: LogicalPlan, rules: &[Box<dyn Rule>]) -> LogicalPlan {
        let mut changed = true;

        // initialize arena
        let mut expr_arena = Arena::new();
        let mut lp_arena = Arena::new();
        let lp_top = to_alp(logical_plan, &mut expr_arena, &mut lp_arena);

        let mut plans = Vec::new();

        // nodes of expressions and lp node from which the expressions are a member of
        let mut exprs = Vec::new();

        // run loop until reaching fixed point
        while changed {
            // recurse into sub plans and expressions and apply rules
            changed = false;
            plans.push(lp_top);
            while let Some(current_node) = plans.pop() {
                // apply rules
                for rule in rules.iter() {
                    // keep iterating over same rule
                    while let Some(x) = rule.optimize_plan(&lp_arena, &lp_arena.get(current_node)) {
                        lp_arena.assign(current_node, x);
                        changed = true;
                    }
                }

                let plan = lp_arena.get(current_node);

                // traverse subplans and expressions and add to the stack
                match plan {
                    ALogicalPlan::Slice { input, .. } => {
                        plans.push(*input);
                    }
                    ALogicalPlan::Selection { input, predicate } => {
                        plans.push(*input);
                        exprs.push((*predicate, *input));
                    }
                    ALogicalPlan::Projection { expr, input, .. } => {
                        plans.push(*input);
                        exprs.extend(expr.iter().map(|e| (*e, *input)));
                    }
                    ALogicalPlan::LocalProjection { expr, input, .. } => {
                        plans.push(*input);
                        exprs.extend(expr.iter().map(|e| (*e, *input)));
                    }
                    ALogicalPlan::Sort { input, .. } => {
                        plans.push(*input);
                    }
                    ALogicalPlan::Explode { input, .. } => {
                        plans.push(*input);
                    }
                    ALogicalPlan::Cache { input } => {
                        plans.push(*input);
                    }
                    ALogicalPlan::Aggregate { input, aggs, .. } => {
                        plans.push(*input);
                        exprs.extend(aggs.iter().map(|e| (*e, *input)));
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
                        exprs.extend(e2.iter().map(|e| (*e, *input)));
                    }
                    ALogicalPlan::Distinct { input, .. } => plans.push(*input),
                    ALogicalPlan::DataFrameScan { selection, .. } => {
                        if let Some(selection) = *selection {
                            exprs.push((selection, current_node))
                        }
                    }
                    ALogicalPlan::CsvScan { predicate, .. } => {
                        if let Some(predicate) = *predicate {
                            exprs.push((predicate, current_node))
                        }
                    }
                    #[cfg(feature = "parquet")]
                    ALogicalPlan::ParquetScan { predicate, .. } => {
                        if let Some(predicate) = *predicate {
                            exprs.push((predicate, current_node))
                        }
                    }
                }

                // process the expressions on the stack and apply optimizations.
                while let Some((current_expr_node, current_lp_node)) = exprs.pop() {
                    for rule in rules.iter() {
                        // keep iterating over same rule
                        while let Some(x) = rule.optimize_expr(
                            &mut expr_arena,
                            current_expr_node,
                            &lp_arena,
                            current_lp_node,
                        ) {
                            expr_arena.assign(current_expr_node, x);
                            changed = true;
                        }
                    }

                    // traverse subexpressions and add to the stack
                    let expr = expr_arena.get(current_expr_node);

                    match expr {
                        AExpr::Duplicated(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::Unique(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::Reverse(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::BinaryExpr { left, right, .. } => {
                            exprs.push((*left, current_lp_node));
                            exprs.push((*right, current_lp_node));
                        }
                        AExpr::Alias(expr, ..) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::Not(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::IsNotNull(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::IsNull(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::Cast { expr, .. } => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::Sort { expr, .. } => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::Min(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::Max(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::Median(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::NUnique(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::First(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::Last(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::List(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::Mean(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::Quantile { expr, .. } => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::Sum(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::AggGroups(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::Shift { input, .. } => {
                            exprs.push((*input, current_lp_node));
                        }
                        AExpr::Count(expr) => {
                            exprs.push((*expr, current_lp_node));
                        }
                        AExpr::Ternary {
                            predicate,
                            truthy,
                            falsy,
                        } => {
                            exprs.push((*predicate, current_lp_node));
                            exprs.push((*truthy, current_lp_node));
                            exprs.push((*falsy, current_lp_node));
                        }
                        AExpr::Apply { input, .. } => {
                            exprs.push((*input, current_lp_node));
                        }
                        AExpr::Window {
                            function,
                            partition_by,
                            order_by,
                        } => {
                            exprs.push((*function, current_lp_node));
                            exprs.push((*partition_by, current_lp_node));
                            if let Some(order_by) = order_by {
                                exprs.push((*order_by, current_lp_node))
                            }
                        }
                        AExpr::Literal { .. } | AExpr::Column { .. } | AExpr::Wildcard => {}
                    }
                }
            }
        }

        node_to_lp(lp_top, &mut expr_arena, &mut lp_arena)
    }
}

// AExpr representation of Nodes which are allocated in an Arena
#[derive(Clone)]
pub enum AExpr {
    Unique(Node),
    Duplicated(Node),
    Reverse(Node),
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
    Min(Node),
    Max(Node),
    Median(Node),
    NUnique(Node),
    First(Node),
    Last(Node),
    Mean(Node),
    List(Node),
    Quantile {
        expr: Node,
        quantile: f64,
    },
    Sum(Node),
    Count(Node),
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
    Window {
        function: Node,
        partition_by: Node,
        order_by: Option<Node>,
    },
    Wildcard,
}

impl Default for AExpr {
    fn default() -> Self {
        AExpr::Wildcard
    }
}

impl AExpr {
    /// This should be a 1 on 1 copy of the get_type method of Expr until Expr is completely phased out.
    pub(crate) fn get_type(&self, schema: &Schema, arena: &Arena<AExpr>) -> Result<ArrowDataType> {
        use AExpr::*;
        match self {
            Window { function, .. } => arena.get(*function).get_type(schema, arena),
            Unique(_) => Ok(ArrowDataType::Boolean),
            Duplicated(_) => Ok(ArrowDataType::Boolean),
            Reverse(expr) => arena.get(*expr).get_type(schema, arena),
            Alias(expr, ..) => arena.get(*expr).get_type(schema, arena),
            Column(name) => Ok(schema.field_with_name(name)?.data_type().clone()),
            Literal(sv) => Ok(sv.get_datatype()),
            BinaryExpr { left, op, right } => match op {
                Operator::Not
                | Operator::Lt
                | Operator::Gt
                | Operator::Eq
                | Operator::NotEq
                | Operator::And
                | Operator::LtEq
                | Operator::GtEq
                | Operator::Or
                | Operator::NotLike
                | Operator::Like => Ok(ArrowDataType::Boolean),
                _ => {
                    let left_type = arena.get(*left).get_type(schema, arena)?;
                    let right_type = arena.get(*right).get_type(schema, arena)?;
                    get_supertype(&left_type, &right_type)
                }
            },
            Not(_) => Ok(ArrowDataType::Boolean),
            IsNull(_) => Ok(ArrowDataType::Boolean),
            IsNotNull(_) => Ok(ArrowDataType::Boolean),
            Sort { expr, .. } => arena.get(*expr).get_type(schema, arena),
            Min(expr) => arena.get(*expr).get_type(schema, arena),
            Max(expr) => arena.get(*expr).get_type(schema, arena),
            Sum(expr) => arena.get(*expr).get_type(schema, arena),
            First(expr) => arena.get(*expr).get_type(schema, arena),
            Last(expr) => arena.get(*expr).get_type(schema, arena),
            Count(expr) => arena.get(*expr).get_type(schema, arena),
            List(expr) => Ok(ArrowDataType::List(Box::new(
                arena.get(*expr).get_type(schema, arena)?,
            ))),
            Mean(expr) => arena.get(*expr).get_type(schema, arena),
            Median(expr) => arena.get(*expr).get_type(schema, arena),
            AggGroups(_) => Ok(ArrowDataType::List(Box::new(ArrowDataType::UInt32))),
            NUnique(_) => Ok(ArrowDataType::UInt32),
            Quantile { expr, .. } => arena.get(*expr).get_type(schema, arena),
            Cast { data_type, .. } => Ok(data_type.clone()),
            Ternary { truthy, .. } => arena.get(*truthy).get_type(schema, arena),
            Apply {
                input, output_type, ..
            } => match output_type {
                Some(output_type) => Ok(output_type.clone()),
                None => arena.get(*input).get_type(schema, arena),
            },
            Shift { input, .. } => arena.get(*input).get_type(schema, arena),
            Wildcard => panic!("should be no wildcard at this point"),
        }
    }
}

// ALogicalPlan is a representation of LogicalPlan with Nodes which are allocated in an Arena
pub enum ALogicalPlan {
    Slice {
        input: Node,
        offset: usize,
        len: usize,
    },
    Selection {
        input: Node,
        predicate: Node,
    },
    CsvScan {
        path: String,
        schema: SchemaRef,
        has_header: bool,
        delimiter: u8,
        ignore_errors: bool,
        skip_rows: usize,
        stop_after_n_rows: Option<usize>,
        with_columns: Option<Vec<String>>,
        predicate: Option<Node>,
        cache: bool,
    },
    #[cfg(feature = "parquet")]
    ParquetScan {
        path: String,
        schema: Schema,
        with_columns: Option<Vec<String>>,
        predicate: Option<Node>,
        stop_after_n_rows: Option<usize>,
        cache: bool,
    },
    DataFrameScan {
        df: Arc<DataFrame>,
        schema: Schema,
        projection: Option<Vec<Node>>,
        selection: Option<Node>,
    },
    Projection {
        expr: Vec<Node>,
        input: Node,
        schema: Schema,
    },
    LocalProjection {
        expr: Vec<Node>,
        input: Node,
        schema: Schema,
    },
    Sort {
        input: Node,
        by_column: String,
        reverse: bool,
    },
    Explode {
        input: Node,
        column: String,
    },
    Cache {
        input: Node,
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
        allow_par: bool,
        force_par: bool,
    },
    HStack {
        input: Node,
        exprs: Vec<Node>,
        schema: Schema,
    },
    Distinct {
        input: Node,
        maintain_order: bool,
        subset: Arc<Option<Vec<String>>>,
    },
}

impl Default for ALogicalPlan {
    fn default() -> Self {
        ALogicalPlan::Selection {
            input: Node(0),
            predicate: Node(0),
        }
    }
}

impl ALogicalPlan {
    pub(crate) fn schema<'a>(&'a self, arena: &'a Arena<ALogicalPlan>) -> &'a Schema {
        use ALogicalPlan::*;
        match self {
            Cache { input } => arena.get(*input).schema(arena),
            Sort { input, .. } => arena.get(*input).schema(arena),
            Explode { input, .. } => arena.get(*input).schema(arena),
            #[cfg(feature = "parquet")]
            ParquetScan { schema, .. } => schema,
            DataFrameScan { schema, .. } => schema,
            Selection { input, .. } => arena.get(*input).schema(arena),
            CsvScan { schema, .. } => schema,
            Projection { schema, .. } => schema,
            LocalProjection { schema, .. } => schema,
            Aggregate { schema, .. } => schema,
            Join { schema, .. } => schema,
            HStack { schema, .. } => schema,
            Distinct { input, .. } => arena.get(*input).schema(arena),
            Slice { input, .. } => arena.get(*input).schema(arena),
        }
    }
}

// converts expression to AExpr, which uses an arena (Vec) for allocation
fn to_aexpr(expr: Expr, arena: &mut Arena<AExpr>) -> Node {
    let v = match expr {
        Expr::Unique(expr) => AExpr::Unique(to_aexpr(*expr, arena)),
        Expr::Duplicated(expr) => AExpr::Duplicated(to_aexpr(*expr, arena)),
        Expr::Reverse(expr) => AExpr::Reverse(to_aexpr(*expr, arena)),
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
        Expr::Min(expr) => AExpr::Min(to_aexpr(*expr, arena)),
        Expr::Max(expr) => AExpr::Max(to_aexpr(*expr, arena)),
        Expr::Median(expr) => AExpr::Median(to_aexpr(*expr, arena)),
        Expr::NUnique(expr) => AExpr::NUnique(to_aexpr(*expr, arena)),
        Expr::First(expr) => AExpr::First(to_aexpr(*expr, arena)),
        Expr::Last(expr) => AExpr::Last(to_aexpr(*expr, arena)),
        Expr::Mean(expr) => AExpr::Mean(to_aexpr(*expr, arena)),
        Expr::List(expr) => AExpr::List(to_aexpr(*expr, arena)),
        Expr::Count(expr) => AExpr::Count(to_aexpr(*expr, arena)),
        Expr::Quantile { expr, quantile } => AExpr::Quantile {
            expr: to_aexpr(*expr, arena),
            quantile,
        },
        Expr::Sum(expr) => AExpr::Sum(to_aexpr(*expr, arena)),
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
        Expr::Window {
            function,
            partition_by,
            order_by,
        } => AExpr::Window {
            function: to_aexpr(*function, arena),
            partition_by: to_aexpr(*partition_by, arena),
            order_by: order_by.map(|ob| to_aexpr(*ob, arena)),
        },
        Expr::Wildcard => AExpr::Wildcard,
    };
    arena.add(v)
}

fn to_alp(
    lp: LogicalPlan,
    expr_arena: &mut Arena<AExpr>,
    lp_arena: &mut Arena<ALogicalPlan>,
) -> Node {
    let v = match lp {
        LogicalPlan::Selection { input, predicate } => {
            let i = to_alp(*input, expr_arena, lp_arena);
            let p = to_aexpr(predicate, expr_arena);
            ALogicalPlan::Selection {
                input: i,
                predicate: p,
            }
        }
        LogicalPlan::Slice { input, offset, len } => {
            let input = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::Slice { input, offset, len }
        }
        LogicalPlan::CsvScan {
            path,
            schema,
            has_header,
            delimiter,
            ignore_errors,
            skip_rows,
            stop_after_n_rows,
            with_columns,
            predicate,
            cache,
        } => ALogicalPlan::CsvScan {
            path,
            schema,
            has_header,
            delimiter,
            ignore_errors,
            skip_rows,
            stop_after_n_rows,
            with_columns,
            predicate: predicate.map(|expr| to_aexpr(expr, expr_arena)),
            cache,
        },
        #[cfg(feature = "parquet")]
        LogicalPlan::ParquetScan {
            path,
            schema,
            with_columns,
            predicate,
            stop_after_n_rows,
            cache,
        } => ALogicalPlan::ParquetScan {
            path,
            schema,
            with_columns,
            predicate: predicate.map(|expr| to_aexpr(expr, expr_arena)),
            stop_after_n_rows,
            cache,
        },
        LogicalPlan::DataFrameScan {
            df,
            schema,
            projection,
            selection,
        } => ALogicalPlan::DataFrameScan {
            df,
            schema,
            projection: projection
                .map(|exprs| exprs.into_iter().map(|x| to_aexpr(x, expr_arena)).collect()),
            selection: selection.map(|expr| to_aexpr(expr, expr_arena)),
        },
        LogicalPlan::Projection {
            expr,
            input,
            schema,
        } => {
            let exp = expr.into_iter().map(|x| to_aexpr(x, expr_arena)).collect();
            let i = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::Projection {
                expr: exp,
                input: i,
                schema,
            }
        }
        LogicalPlan::LocalProjection {
            expr,
            input,
            schema,
        } => {
            let exp = expr.into_iter().map(|x| to_aexpr(x, expr_arena)).collect();
            let i = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::LocalProjection {
                expr: exp,
                input: i,
                schema,
            }
        }
        LogicalPlan::Sort {
            input,
            by_column,
            reverse,
        } => {
            let input = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::Sort {
                input,
                by_column,
                reverse,
            }
        }
        LogicalPlan::Explode { input, column } => {
            let input = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::Explode { input, column }
        }
        LogicalPlan::Cache { input } => {
            let input = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::Cache { input }
        }
        LogicalPlan::Aggregate {
            input,
            keys,
            aggs,
            schema,
        } => {
            let i = to_alp(*input, expr_arena, lp_arena);
            let aggs_new = aggs.into_iter().map(|x| to_aexpr(x, expr_arena)).collect();

            ALogicalPlan::Aggregate {
                input: i,
                keys,
                aggs: aggs_new,
                schema,
            }
        }
        LogicalPlan::Join {
            input_left,
            input_right,
            schema,
            how,
            left_on,
            right_on,
            allow_par,
            force_par,
        } => {
            let i_l = to_alp(*input_left, expr_arena, lp_arena);
            let i_r = to_alp(*input_right, expr_arena, lp_arena);

            let l_on = to_aexpr(left_on, expr_arena);

            let r_on = to_aexpr(right_on, expr_arena);

            ALogicalPlan::Join {
                input_left: i_l,
                input_right: i_r,
                schema,
                left_on: l_on,
                how,
                right_on: r_on,
                allow_par,
                force_par,
            }
        }
        LogicalPlan::HStack {
            input,
            exprs,
            schema,
        } => {
            let exp = exprs.into_iter().map(|x| to_aexpr(x, expr_arena)).collect();
            let i = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::HStack {
                input: i,
                exprs: exp,
                schema,
            }
        }
        LogicalPlan::Distinct {
            input,
            maintain_order,
            subset,
        } => {
            let i = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::Distinct {
                input: i,
                maintain_order,
                subset,
            }
        }
    };
    lp_arena.add(v)
}

fn node_to_exp(node: Node, expr_arena: &mut Arena<AExpr>) -> Expr {
    let expr = expr_arena.get_mut(node);
    let expr = std::mem::take(expr);

    match expr {
        AExpr::Duplicated(node) => Expr::Duplicated(Box::new(node_to_exp(node, expr_arena))),
        AExpr::Unique(node) => Expr::Unique(Box::new(node_to_exp(node, expr_arena))),
        AExpr::Reverse(node) => Expr::Reverse(Box::new(node_to_exp(node, expr_arena))),
        AExpr::Alias(expr, name) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Alias(Box::new(exp), name)
        }
        AExpr::Column(a) => Expr::Column(a),
        AExpr::Literal(s) => Expr::Literal(s),
        AExpr::BinaryExpr { left, op, right } => {
            let l = node_to_exp(left, expr_arena);
            let r = node_to_exp(right, expr_arena);
            Expr::BinaryExpr {
                left: Box::new(l),
                op,
                right: Box::new(r),
            }
        }
        AExpr::Not(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Not(Box::new(exp))
        }
        AExpr::IsNotNull(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::IsNotNull(Box::new(exp))
        }
        AExpr::IsNull(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::IsNull(Box::new(exp))
        }
        AExpr::Cast { expr, data_type } => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Cast {
                expr: Box::new(exp),
                data_type,
            }
        }
        AExpr::Sort { expr, reverse } => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Sort {
                expr: Box::new(exp),
                reverse,
            }
        }
        AExpr::Min(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Min(Box::new(exp))
        }
        AExpr::Max(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Max(Box::new(exp))
        }

        AExpr::Median(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Median(Box::new(exp))
        }
        AExpr::NUnique(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::NUnique(Box::new(exp))
        }
        AExpr::First(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::First(Box::new(exp))
        }
        AExpr::Last(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Last(Box::new(exp))
        }
        AExpr::Mean(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Mean(Box::new(exp))
        }
        AExpr::List(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::List(Box::new(exp))
        }
        AExpr::Quantile { expr, quantile } => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Quantile {
                expr: Box::new(exp),
                quantile,
            }
        }
        AExpr::Sum(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Sum(Box::new(exp))
        }
        AExpr::AggGroups(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::AggGroups(Box::new(exp))
        }
        AExpr::Shift { input, periods } => {
            let e = node_to_exp(input, expr_arena);
            Expr::Shift {
                input: Box::new(e),
                periods,
            }
        }
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            let p = node_to_exp(predicate, expr_arena);
            let t = node_to_exp(truthy, expr_arena);
            let f = node_to_exp(falsy, expr_arena);

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
            let i = node_to_exp(input, expr_arena);
            Expr::Apply {
                input: Box::new(i),
                function,
                output_type,
            }
        }
        AExpr::Count(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Count(Box::new(exp))
        }
        AExpr::Window {
            function,
            partition_by,
            order_by,
        } => {
            let function = Box::new(node_to_exp(function, expr_arena));
            let partition_by = Box::new(node_to_exp(partition_by, expr_arena));
            let order_by = order_by.map(|ob| Box::new(node_to_exp(ob, expr_arena)));
            Expr::Window {
                function,
                partition_by,
                order_by,
            }
        }
        AExpr::Wildcard => Expr::Wildcard,
    }
}

fn node_to_lp(
    node: Node,
    expr_arena: &mut Arena<AExpr>,
    lp_arena: &mut Arena<ALogicalPlan>,
) -> LogicalPlan {
    let lp = lp_arena.get_mut(node);
    let lp = std::mem::take(lp);

    match lp {
        ALogicalPlan::Slice { input, offset, len } => {
            let lp = node_to_lp(input, expr_arena, lp_arena);
            LogicalPlan::Slice {
                input: Box::new(lp),
                offset,
                len,
            }
        }
        ALogicalPlan::Selection { input, predicate } => {
            let lp = node_to_lp(input, expr_arena, lp_arena);
            let p = node_to_exp(predicate, expr_arena);
            LogicalPlan::Selection {
                input: Box::new(lp),
                predicate: p,
            }
        }
        ALogicalPlan::CsvScan {
            path,
            schema,
            has_header,
            delimiter,
            ignore_errors,
            skip_rows,
            stop_after_n_rows,
            with_columns,
            predicate,
            cache,
        } => LogicalPlan::CsvScan {
            path,
            schema,
            has_header,
            delimiter,
            ignore_errors,
            skip_rows,
            stop_after_n_rows,
            with_columns,
            predicate: predicate.map(|n| node_to_exp(n, expr_arena)),
            cache,
        },
        #[cfg(feature = "parquet")]
        ALogicalPlan::ParquetScan {
            path,
            schema,
            with_columns,
            predicate,
            stop_after_n_rows,
            cache,
        } => LogicalPlan::ParquetScan {
            path,
            schema,
            with_columns,
            predicate: predicate.map(|n| node_to_exp(n, expr_arena)),
            stop_after_n_rows,
            cache,
        },
        ALogicalPlan::DataFrameScan {
            df,
            schema,
            projection,
            selection,
        } => LogicalPlan::DataFrameScan {
            df,
            schema,
            projection: projection
                .as_ref()
                .map(|nodes| nodes.iter().map(|n| node_to_exp(*n, expr_arena)).collect()),
            selection: selection.map(|n| node_to_exp(n, expr_arena)),
        },
        ALogicalPlan::Projection {
            expr,
            input,
            schema,
        } => {
            let exprs = expr.iter().map(|x| node_to_exp(*x, expr_arena)).collect();
            let i = node_to_lp(input, expr_arena, lp_arena);

            LogicalPlan::Projection {
                expr: exprs,
                input: Box::new(i),
                schema,
            }
        }
        ALogicalPlan::LocalProjection {
            expr,
            input,
            schema,
        } => {
            let exprs = expr.iter().map(|x| node_to_exp(*x, expr_arena)).collect();
            let i = node_to_lp(input, expr_arena, lp_arena);

            LogicalPlan::LocalProjection {
                expr: exprs,
                input: Box::new(i),
                schema,
            }
        }
        ALogicalPlan::Sort {
            input,
            by_column,
            reverse,
        } => {
            let input = Box::new(node_to_lp(input, expr_arena, lp_arena));
            LogicalPlan::Sort {
                input,
                by_column,
                reverse,
            }
        }
        ALogicalPlan::Explode { input, column } => {
            let input = Box::new(node_to_lp(input, expr_arena, lp_arena));
            LogicalPlan::Explode { input, column }
        }
        ALogicalPlan::Cache { input } => {
            let input = Box::new(node_to_lp(input, expr_arena, lp_arena));
            LogicalPlan::Cache { input }
        }
        ALogicalPlan::Aggregate {
            input,
            keys,
            aggs,
            schema,
        } => {
            let i = node_to_lp(input, expr_arena, lp_arena);
            let a = aggs.iter().map(|x| node_to_exp(*x, expr_arena)).collect();

            LogicalPlan::Aggregate {
                input: Box::new(i),
                keys,
                aggs: a,
                schema,
            }
        }
        ALogicalPlan::Join {
            input_left,
            input_right,
            schema,
            how,
            left_on,
            right_on,
            allow_par,
            force_par,
        } => {
            let i_l = node_to_lp(input_left, expr_arena, lp_arena);
            let i_r = node_to_lp(input_right, expr_arena, lp_arena);

            let l_on = node_to_exp(left_on, expr_arena);
            let r_on = node_to_exp(right_on, expr_arena);

            LogicalPlan::Join {
                input_left: Box::new(i_l),
                input_right: Box::new(i_r),
                schema,
                how,
                left_on: l_on,
                right_on: r_on,
                allow_par,
                force_par,
            }
        }
        ALogicalPlan::HStack {
            input,
            exprs,
            schema,
        } => {
            let i = node_to_lp(input, expr_arena, lp_arena);
            let e = exprs.iter().map(|x| node_to_exp(*x, expr_arena)).collect();

            LogicalPlan::HStack {
                input: Box::new(i),
                exprs: e,
                schema,
            }
        }
        ALogicalPlan::Distinct {
            input,
            maintain_order,
            subset,
        } => {
            let i = node_to_lp(input, expr_arena, lp_arena);
            LogicalPlan::Distinct {
                input: Box::new(i),
                maintain_order,
                subset,
            }
        }
    }
}

pub trait Rule {
    fn optimize_plan(
        &self,
        _arena: &Arena<ALogicalPlan>,
        _logical_plan: &ALogicalPlan,
    ) -> Option<ALogicalPlan> {
        None
    }
    fn optimize_expr(
        &self,
        _expr_arena: &mut Arena<AExpr>,
        _expr_node: Node,
        _lp_arena: &Arena<ALogicalPlan>,
        _lp_node: Node,
    ) -> Option<AExpr> {
        None
    }
}
