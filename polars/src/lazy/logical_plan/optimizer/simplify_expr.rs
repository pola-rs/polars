use crate::lazy::logical_plan::*;
use crate::lazy::prelude::*;
use crate::prelude::*;
use crate::utils::{Arena, Node};
use std::sync::Arc;

// AExpr representation of Nodes which are allocated in an Arena
#[derive(Clone)]
enum AExpr {
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
    AggCount(Node),
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

// ALogicalPlan is a representation of LogicalPlan with Nodes which are allocated in an Arena
enum ALogicalPlan {
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
        Expr::Min(expr) => AExpr::AggMin(to_aexpr(*expr, arena)),
        Expr::Max(expr) => AExpr::AggMax(to_aexpr(*expr, arena)),
        Expr::Median(expr) => AExpr::AggMedian(to_aexpr(*expr, arena)),
        Expr::NUnique(expr) => AExpr::AggNUnique(to_aexpr(*expr, arena)),
        Expr::First(expr) => AExpr::AggFirst(to_aexpr(*expr, arena)),
        Expr::Last(expr) => AExpr::AggLast(to_aexpr(*expr, arena)),
        Expr::Mean(expr) => AExpr::AggMean(to_aexpr(*expr, arena)),
        Expr::List(expr) => AExpr::AggList(to_aexpr(*expr, arena)),
        Expr::Count(expr) => AExpr::AggCount(to_aexpr(*expr, arena)),
        Expr::Quantile { expr, quantile } => AExpr::AggQuantile {
            expr: to_aexpr(*expr, arena),
            quantile,
        },
        Expr::Sum(expr) => AExpr::AggSum(to_aexpr(*expr, arena)),
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
        AExpr::AggMin(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Min(Box::new(exp))
        }
        AExpr::AggMax(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Max(Box::new(exp))
        }

        AExpr::AggMedian(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Median(Box::new(exp))
        }
        AExpr::AggNUnique(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::NUnique(Box::new(exp))
        }
        AExpr::AggFirst(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::First(Box::new(exp))
        }
        AExpr::AggLast(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Last(Box::new(exp))
        }
        AExpr::AggMean(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Mean(Box::new(exp))
        }
        AExpr::AggList(expr) => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::List(Box::new(exp))
        }
        AExpr::AggQuantile { expr, quantile } => {
            let exp = node_to_exp(expr, expr_arena);
            Expr::Quantile {
                expr: Box::new(exp),
                quantile,
            }
        }
        AExpr::AggSum(expr) => {
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
        AExpr::AggCount(expr) => {
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
                    ALogicalPlan::Slice { input, .. } => {
                        plans.push(*input);
                    }
                    ALogicalPlan::Selection { input, predicate } => {
                        plans.push(*input);
                        exprs.push(*predicate);
                    }
                    ALogicalPlan::Projection { expr, input, .. } => {
                        plans.push(*input);
                        exprs.extend(expr);
                    }
                    ALogicalPlan::LocalProjection { expr, input, .. } => {
                        plans.push(*input);
                        exprs.extend(expr);
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
                    ALogicalPlan::Distinct { input, .. } => plans.push(*input),
                    ALogicalPlan::DataFrameScan { selection, .. } => {
                        if let Some(selection) = *selection {
                            exprs.push(selection)
                        }
                    }
                    ALogicalPlan::CsvScan { predicate, .. } => {
                        if let Some(predicate) = *predicate {
                            exprs.push(predicate)
                        }
                    }
                    #[cfg(feature = "parquet")]
                    ALogicalPlan::ParquetScan { predicate, .. } => {
                        if let Some(predicate) = *predicate {
                            exprs.push(predicate)
                        }
                    }
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
                        AExpr::Duplicated(expr) => {
                            exprs.push(*expr);
                        }
                        AExpr::Unique(expr) => {
                            exprs.push(*expr);
                        }
                        AExpr::Reverse(expr) => {
                            exprs.push(*expr);
                        }
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
                        AExpr::AggCount(expr) => {
                            exprs.push(*expr);
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
                        AExpr::Window {
                            function,
                            partition_by,
                            order_by,
                        } => {
                            exprs.push(*function);
                            exprs.push(*partition_by);
                            if let Some(order_by) = order_by {
                                exprs.push(*order_by)
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
