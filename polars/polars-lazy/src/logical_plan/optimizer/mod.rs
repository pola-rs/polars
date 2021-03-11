use std::collections::{HashMap, HashSet};

use ahash::RandomState;

use polars_core::frame::group_by::{fmt_groupby_column, GroupByMethod};
use polars_core::frame::hash_join::JoinType;
use polars_core::prelude::*;
use polars_core::utils::{get_supertype, Arena, Node};

use crate::logical_plan::{det_melt_schema, Context};
use crate::prelude::*;
use crate::utils::{aexprs_to_schema, rename_field};

pub(crate) mod aggregate_pushdown;
pub(crate) mod aggregate_scan_projections;
pub(crate) mod predicate_pushdown;
pub(crate) mod projection_pushdown;
pub(crate) mod simplify_expr;
pub(crate) mod type_coercion;

pub trait Optimize {
    fn optimize(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan>;
}

// arbitrary constant to reduce reallocation.
// don't expect more than 100 predicates.
const HASHMAP_SIZE: usize = 100;

pub(crate) fn init_hashmap<K, V>() -> HashMap<K, V, RandomState> {
    HashMap::with_capacity_and_hasher(HASHMAP_SIZE, RandomState::new())
}

/// Optimizer that uses a stack and memory arenas in favor of recursion
pub struct StackOptimizer {}

impl StackOptimizer {
    pub fn optimize_loop(
        &self,
        rules: &mut [Box<dyn OptimizationRule>],
        expr_arena: &mut Arena<AExpr>,
        lp_arena: &mut Arena<ALogicalPlan>,
        lp_top: Node,
    ) -> Node {
        let mut changed = true;

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
                for rule in rules.iter_mut() {
                    // keep iterating over same rule
                    while let Some(x) = rule.optimize_plan(lp_arena, expr_arena, current_node) {
                        lp_arena.replace(current_node, x);
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
                    ALogicalPlan::Aggregate {
                        input, aggs, keys, ..
                    } => {
                        plans.push(*input);
                        exprs.extend(aggs.iter().map(|e| (*e, *input)));
                        exprs.extend(keys.iter().map(|e| (*e, *input)));
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
                    ALogicalPlan::Melt { input, .. } => plans.push(*input),
                    ALogicalPlan::Udf { input, .. } => plans.push(*input),
                }

                // process the expressions on the stack and apply optimizations.
                while let Some((current_expr_node, current_lp_node)) = exprs.pop() {
                    for rule in rules.iter() {
                        // keep iterating over same rule
                        while let Some(x) = rule.optimize_expr(
                            expr_arena,
                            current_expr_node,
                            &lp_arena,
                            current_lp_node,
                        ) {
                            expr_arena.replace(current_expr_node, x);
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
                        AExpr::Explode(expr) => {
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
                        AExpr::Agg(agg) => match agg {
                            AAggExpr::Min(expr) => {
                                exprs.push((*expr, current_lp_node));
                            }
                            AAggExpr::Max(expr) => {
                                exprs.push((*expr, current_lp_node));
                            }
                            AAggExpr::Median(expr) => {
                                exprs.push((*expr, current_lp_node));
                            }
                            AAggExpr::NUnique(expr) => {
                                exprs.push((*expr, current_lp_node));
                            }
                            AAggExpr::First(expr) => {
                                exprs.push((*expr, current_lp_node));
                            }
                            AAggExpr::Last(expr) => {
                                exprs.push((*expr, current_lp_node));
                            }
                            AAggExpr::List(expr) => {
                                exprs.push((*expr, current_lp_node));
                            }
                            AAggExpr::Mean(expr) => {
                                exprs.push((*expr, current_lp_node));
                            }
                            AAggExpr::Quantile { expr, .. } => {
                                exprs.push((*expr, current_lp_node));
                            }
                            AAggExpr::Sum(expr) => {
                                exprs.push((*expr, current_lp_node));
                            }
                            AAggExpr::AggGroups(expr) => {
                                exprs.push((*expr, current_lp_node));
                            }
                            AAggExpr::Count(expr) => {
                                exprs.push((*expr, current_lp_node));
                            }
                            AAggExpr::Std(expr) => {
                                exprs.push((*expr, current_lp_node));
                            }
                            AAggExpr::Var(expr) => {
                                exprs.push((*expr, current_lp_node));
                            }
                        },
                        AExpr::Shift { input, .. } => {
                            exprs.push((*input, current_lp_node));
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
                        AExpr::Udf { input, .. } => {
                            exprs.push((*input, current_lp_node));
                        }
                        AExpr::BinaryFunction {
                            input_a, input_b, ..
                        } => {
                            exprs.push((*input_a, current_lp_node));
                            exprs.push((*input_b, current_lp_node));
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
                        AExpr::Slice { input, .. } => {
                            exprs.push((*input, current_lp_node));
                        }
                        AExpr::Literal { .. } | AExpr::Column { .. } | AExpr::Wildcard => {}
                    }
                }
            }
        }
        lp_top
    }
}

#[derive(Clone)]
pub enum AAggExpr {
    Min(Node),
    Max(Node),
    Median(Node),
    NUnique(Node),
    First(Node),
    Last(Node),
    Mean(Node),
    List(Node),
    Quantile { expr: Node, quantile: f64 },
    Sum(Node),
    Count(Node),
    Std(Node),
    Var(Node),
    AggGroups(Node),
}

// AExpr representation of Nodes which are allocated in an Arena
#[derive(Clone)]
pub enum AExpr {
    Unique(Node),
    Duplicated(Node),
    Reverse(Node),
    Explode(Node),
    Alias(Node, Arc<String>),
    Column(Arc<String>),
    Literal(LiteralValue),
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
        data_type: DataType,
    },
    Sort {
        expr: Node,
        reverse: bool,
    },
    Agg(AAggExpr),
    Ternary {
        predicate: Node,
        truthy: Node,
        falsy: Node,
    },
    Udf {
        input: Node,
        function: Arc<dyn SeriesUdf>,
        output_type: Option<DataType>,
    },
    Shift {
        input: Node,
        periods: i64,
    },
    Window {
        function: Node,
        partition_by: Node,
        order_by: Option<Node>,
    },
    Wildcard,
    Slice {
        input: Node,
        offset: isize,
        length: usize,
    },
    BinaryFunction {
        input_a: Node,
        input_b: Node,
        function: Arc<dyn SeriesBinaryUdf>,
        /// Delays output type evaluation until input schema is known.
        output_field: Arc<dyn BinaryUdfOutputField>,
    },
}

impl Default for AExpr {
    fn default() -> Self {
        AExpr::Wildcard
    }
}

pub(crate) fn field_by_context(
    mut field: Field,
    ctxt: Context,
    groupby_method: GroupByMethod,
) -> Field {
    if &DataType::Boolean == field.data_type() {
        field = Field::new(field.name(), DataType::UInt32)
    }

    match ctxt {
        Context::Other => field,
        Context::Aggregation => {
            let new_name = fmt_groupby_column(field.name(), groupby_method);
            rename_field(&field, &new_name)
        }
    }
}

impl AExpr {
    /// This should be a 1 on 1 copy of the get_type method of Expr until Expr is completely phased out.
    pub(crate) fn get_type(
        &self,
        schema: &Schema,
        ctxt: Context,
        arena: &Arena<AExpr>,
    ) -> Result<DataType> {
        self.to_field(schema, ctxt, arena)
            .map(|f| f.data_type().clone())
    }

    /// Get Field result of the expression. The schema is the input data.
    pub(crate) fn to_field(
        &self,
        schema: &Schema,
        ctxt: Context,
        arena: &Arena<AExpr>,
    ) -> Result<Field> {
        use AExpr::*;
        match self {
            Window { function, .. } => arena.get(*function).to_field(schema, ctxt, arena),
            Unique(expr) => {
                let field = arena.get(*expr).to_field(&schema, ctxt, arena)?;
                Ok(Field::new(field.name(), DataType::Boolean))
            }
            Duplicated(expr) => {
                let field = arena.get(*expr).to_field(&schema, ctxt, arena)?;
                Ok(Field::new(field.name(), DataType::Boolean))
            }
            Reverse(expr) => arena.get(*expr).to_field(&schema, ctxt, arena),
            Explode(expr) => arena.get(*expr).to_field(&schema, ctxt, arena),
            Alias(expr, name) => Ok(Field::new(
                name,
                arena.get(*expr).get_type(schema, ctxt, arena)?,
            )),
            Column(name) => {
                let field = schema.field_with_name(name).map(|f| f.clone())?;
                Ok(field)
            }
            Literal(sv) => Ok(Field::new("lit", sv.get_datatype())),
            BinaryExpr { left, right, op } => {
                let left_type = arena.get(*left).get_type(schema, ctxt, arena)?;
                let right_type = arena.get(*right).get_type(schema, ctxt, arena)?;

                let expr_type = match op {
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
                    | Operator::Like => DataType::Boolean,
                    _ => get_supertype(&left_type, &right_type)?,
                };

                use Operator::*;
                let out_field;
                let out_name = match op {
                    Plus | Minus | Multiply | Divide | Modulus => {
                        out_field = arena.get(*left).to_field(schema, ctxt, arena)?;
                        out_field.name().as_str()
                    }
                    Eq | Lt | GtEq | LtEq => "",
                    _ => "binary_expr",
                };

                Ok(Field::new(out_name, expr_type))
            }
            Not(_) => Ok(Field::new("not", DataType::Boolean)),
            IsNull(_) => Ok(Field::new("is_null", DataType::Boolean)),
            IsNotNull(_) => Ok(Field::new("is_not_null", DataType::Boolean)),
            Sort { expr, .. } => arena.get(*expr).to_field(schema, ctxt, arena),
            Agg(agg) => {
                use AAggExpr::*;
                let field = match agg {
                    Min(expr) => field_by_context(
                        arena.get(*expr).to_field(schema, ctxt, arena)?,
                        ctxt,
                        GroupByMethod::Min,
                    ),
                    Max(expr) => field_by_context(
                        arena.get(*expr).to_field(schema, ctxt, arena)?,
                        ctxt,
                        GroupByMethod::Max,
                    ),
                    Median(expr) => field_by_context(
                        arena.get(*expr).to_field(schema, ctxt, arena)?,
                        ctxt,
                        GroupByMethod::Median,
                    ),
                    Mean(expr) => field_by_context(
                        arena.get(*expr).to_field(schema, ctxt, arena)?,
                        ctxt,
                        GroupByMethod::Mean,
                    ),
                    First(expr) => field_by_context(
                        arena.get(*expr).to_field(schema, ctxt, arena)?,
                        ctxt,
                        GroupByMethod::First,
                    ),
                    Last(expr) => field_by_context(
                        arena.get(*expr).to_field(schema, ctxt, arena)?,
                        ctxt,
                        GroupByMethod::Last,
                    ),
                    List(expr) => field_by_context(
                        arena.get(*expr).to_field(schema, ctxt, arena)?,
                        ctxt,
                        GroupByMethod::List,
                    ),
                    Std(expr) => {
                        let field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        let field = Field::new(field.name(), DataType::Float64);
                        field_by_context(field, ctxt, GroupByMethod::Std)
                    }
                    Var(expr) => {
                        let field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        let field = Field::new(field.name(), DataType::Float64);
                        field_by_context(field, ctxt, GroupByMethod::Var)
                    }
                    NUnique(expr) => {
                        let field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        let field = Field::new(field.name(), DataType::UInt32);
                        match ctxt {
                            Context::Other => field,
                            Context::Aggregation => {
                                let new_name =
                                    fmt_groupby_column(field.name(), GroupByMethod::NUnique);
                                rename_field(&field, &new_name)
                            }
                        }
                    }
                    Sum(expr) => field_by_context(
                        arena.get(*expr).to_field(schema, ctxt, arena)?,
                        ctxt,
                        GroupByMethod::Sum,
                    ),
                    Count(expr) => {
                        let field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        let field = Field::new(field.name(), DataType::UInt32);
                        match ctxt {
                            Context::Other => field,
                            Context::Aggregation => {
                                let new_name =
                                    fmt_groupby_column(field.name(), GroupByMethod::Count);
                                rename_field(&field, &new_name)
                            }
                        }
                    }
                    AggGroups(expr) => {
                        let field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        let new_name = fmt_groupby_column(field.name(), GroupByMethod::Groups);
                        Field::new(&new_name, DataType::List(ArrowDataType::UInt32))
                    }
                    Quantile { expr, quantile } => field_by_context(
                        arena.get(*expr).to_field(schema, ctxt, arena)?,
                        ctxt,
                        GroupByMethod::Quantile(*quantile),
                    ),
                };
                Ok(field)
            }
            Cast { expr, data_type } => {
                let field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                Ok(Field::new(field.name(), data_type.clone()))
            }
            Ternary { truthy, .. } => arena.get(*truthy).to_field(schema, ctxt, arena),
            Udf {
                output_type, input, ..
            } => match output_type {
                None => arena.get(*input).to_field(schema, ctxt, arena),
                Some(output_type) => {
                    let input_field = arena.get(*input).to_field(schema, ctxt, arena)?;
                    Ok(Field::new(input_field.name(), output_type.clone()))
                }
            },
            BinaryFunction {
                input_a,
                input_b,
                output_field,
                ..
            } => {
                let field_a = arena.get(*input_a).to_field(schema, ctxt, arena)?;
                let field_b = arena.get(*input_b).to_field(schema, ctxt, arena)?;
                let out = output_field.get_field(schema, ctxt, &field_a, &field_b);
                Ok(out.unwrap())
            }
            Shift { input, .. } => arena.get(*input).to_field(schema, ctxt, arena),
            Slice { input, .. } => arena.get(*input).to_field(schema, ctxt, arena),
            Wildcard => panic!("should be no wildcard at this point"),
        }
    }
}

// ALogicalPlan is a representation of LogicalPlan with Nodes which are allocated in an Arena
#[derive(Clone, Debug)]
pub enum ALogicalPlan {
    Melt {
        input: Node,
        id_vars: Arc<Vec<String>>,
        value_vars: Arc<Vec<String>>,
        schema: SchemaRef,
    },
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
        aggregate: Vec<Node>,
        cache: bool,
    },
    #[cfg(feature = "parquet")]
    ParquetScan {
        path: String,
        schema: SchemaRef,
        with_columns: Option<Vec<String>>,
        predicate: Option<Node>,
        aggregate: Vec<Node>,
        stop_after_n_rows: Option<usize>,
        cache: bool,
    },
    DataFrameScan {
        df: Arc<DataFrame>,
        schema: SchemaRef,
        projection: Option<Vec<Node>>,
        selection: Option<Node>,
    },
    Projection {
        expr: Vec<Node>,
        input: Node,
        schema: SchemaRef,
    },
    LocalProjection {
        expr: Vec<Node>,
        input: Node,
        schema: SchemaRef,
    },
    Sort {
        input: Node,
        by_column: String,
        reverse: bool,
    },
    Explode {
        input: Node,
        columns: Vec<String>,
    },
    Cache {
        input: Node,
    },
    Aggregate {
        input: Node,
        keys: Vec<Node>,
        aggs: Vec<Node>,
        schema: SchemaRef,
        apply: Option<Arc<dyn DataFrameUdf>>,
    },
    Join {
        input_left: Node,
        input_right: Node,
        schema: SchemaRef,
        how: JoinType,
        left_on: Vec<Node>,
        right_on: Vec<Node>,
        allow_par: bool,
        force_par: bool,
    },
    HStack {
        input: Node,
        exprs: Vec<Node>,
        schema: SchemaRef,
    },
    Distinct {
        input: Node,
        maintain_order: bool,
        subset: Arc<Option<Vec<String>>>,
    },
    Udf {
        input: Node,
        function: Arc<dyn DataFrameUdf>,
        ///  allow predicate pushdown optimizations
        predicate_pd: bool,
        ///  allow projection pushdown optimizations
        projection_pd: bool,
        schema: Option<SchemaRef>,
    },
}

impl Default for ALogicalPlan {
    fn default() -> Self {
        // the lp is should not be valid. By choosing a max value we'll likely panic indicating
        // a programming error early.
        ALogicalPlan::Selection {
            input: Node(usize::max_value()),
            predicate: Node(usize::max_value()),
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
            Melt { schema, .. } => schema,
            Udf { input, schema, .. } => match schema {
                Some(schema) => schema,
                None => arena.get(*input).schema(arena),
            },
        }
    }
}

// converts expression to AExpr, which uses an arena (Vec) for allocation
pub(crate) fn to_aexpr(expr: Expr, arena: &mut Arena<AExpr>) -> Node {
    let v = match expr {
        Expr::Unique(expr) => AExpr::Unique(to_aexpr(*expr, arena)),
        Expr::Duplicated(expr) => AExpr::Duplicated(to_aexpr(*expr, arena)),
        Expr::Reverse(expr) => AExpr::Reverse(to_aexpr(*expr, arena)),
        Expr::Explode(expr) => AExpr::Explode(to_aexpr(*expr, arena)),
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
        Expr::Agg(agg) => {
            let a_agg = match agg {
                AggExpr::Min(expr) => AAggExpr::Min(to_aexpr(*expr, arena)),
                AggExpr::Max(expr) => AAggExpr::Max(to_aexpr(*expr, arena)),
                AggExpr::Median(expr) => AAggExpr::Median(to_aexpr(*expr, arena)),
                AggExpr::NUnique(expr) => AAggExpr::NUnique(to_aexpr(*expr, arena)),
                AggExpr::First(expr) => AAggExpr::First(to_aexpr(*expr, arena)),
                AggExpr::Last(expr) => AAggExpr::Last(to_aexpr(*expr, arena)),
                AggExpr::Mean(expr) => AAggExpr::Mean(to_aexpr(*expr, arena)),
                AggExpr::List(expr) => AAggExpr::List(to_aexpr(*expr, arena)),
                AggExpr::Count(expr) => AAggExpr::Count(to_aexpr(*expr, arena)),
                AggExpr::Quantile { expr, quantile } => AAggExpr::Quantile {
                    expr: to_aexpr(*expr, arena),
                    quantile,
                },
                AggExpr::Sum(expr) => AAggExpr::Sum(to_aexpr(*expr, arena)),
                AggExpr::Std(expr) => AAggExpr::Std(to_aexpr(*expr, arena)),
                AggExpr::Var(expr) => AAggExpr::Var(to_aexpr(*expr, arena)),
                AggExpr::AggGroups(expr) => AAggExpr::AggGroups(to_aexpr(*expr, arena)),
            };
            AExpr::Agg(a_agg)
        }
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
        Expr::Udf {
            input,
            function,
            output_type,
        } => AExpr::Udf {
            input: to_aexpr(*input, arena),
            function,
            output_type,
        },
        Expr::BinaryFunction {
            input_a,
            input_b,
            function,
            output_field,
        } => AExpr::BinaryFunction {
            input_a: to_aexpr(*input_a, arena),
            input_b: to_aexpr(*input_b, arena),
            function,
            output_field,
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
        Expr::Slice {
            input,
            offset,
            length,
        } => AExpr::Slice {
            input: to_aexpr(*input, arena),
            offset,
            length,
        },
        Expr::Wildcard => AExpr::Wildcard,
        Expr::Except(_) => panic!("should be no except expression at this point"),
    };
    arena.add(v)
}

pub(crate) fn to_alp(
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
        LogicalPlan::Melt {
            input,
            id_vars,
            value_vars,
            schema,
        } => {
            let input = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::Melt {
                input,
                id_vars,
                value_vars,
                schema,
            }
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
            aggregate,
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
            aggregate: aggregate
                .into_iter()
                .map(|expr| to_aexpr(expr, expr_arena))
                .collect(),
            cache,
        },
        #[cfg(feature = "parquet")]
        LogicalPlan::ParquetScan {
            path,
            schema,
            with_columns,
            predicate,
            aggregate,
            stop_after_n_rows,
            cache,
        } => ALogicalPlan::ParquetScan {
            path,
            schema,
            with_columns,
            predicate: predicate.map(|expr| to_aexpr(expr, expr_arena)),
            aggregate: aggregate
                .into_iter()
                .map(|expr| to_aexpr(expr, expr_arena))
                .collect(),
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
        LogicalPlan::Explode { input, columns } => {
            let input = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::Explode { input, columns }
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
            apply,
        } => {
            let i = to_alp(*input, expr_arena, lp_arena);
            let aggs_new = aggs.into_iter().map(|x| to_aexpr(x, expr_arena)).collect();
            let keys_new = keys
                .iter()
                .map(|x| to_aexpr(x.clone(), expr_arena))
                .collect();

            ALogicalPlan::Aggregate {
                input: i,
                keys: keys_new,
                aggs: aggs_new,
                schema,
                apply,
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

            let l_on = left_on
                .into_iter()
                .map(|x| to_aexpr(x, expr_arena))
                .collect();
            let r_on = right_on
                .into_iter()
                .map(|x| to_aexpr(x, expr_arena))
                .collect();

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
        LogicalPlan::Udf {
            input,
            function,
            projection_pd,
            predicate_pd,
            schema,
        } => {
            let input = to_alp(*input, expr_arena, lp_arena);
            ALogicalPlan::Udf {
                input,
                function,
                projection_pd,
                predicate_pd,
                schema,
            }
        }
    };
    lp_arena.add(v)
}

pub(crate) fn node_to_exp(node: Node, expr_arena: &Arena<AExpr>) -> Expr {
    let expr = expr_arena.get(node).clone();

    match expr {
        AExpr::Duplicated(node) => Expr::Duplicated(Box::new(node_to_exp(node, expr_arena))),
        AExpr::Unique(node) => Expr::Unique(Box::new(node_to_exp(node, expr_arena))),
        AExpr::Reverse(node) => Expr::Reverse(Box::new(node_to_exp(node, expr_arena))),
        AExpr::Explode(node) => Expr::Explode(Box::new(node_to_exp(node, expr_arena))),
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
        AExpr::Agg(agg) => match agg {
            AAggExpr::Min(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Min(Box::new(exp)).into()
            }
            AAggExpr::Max(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Max(Box::new(exp)).into()
            }

            AAggExpr::Median(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Median(Box::new(exp)).into()
            }
            AAggExpr::NUnique(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::NUnique(Box::new(exp)).into()
            }
            AAggExpr::First(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::First(Box::new(exp)).into()
            }
            AAggExpr::Last(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Last(Box::new(exp)).into()
            }
            AAggExpr::Mean(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Mean(Box::new(exp)).into()
            }
            AAggExpr::List(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::List(Box::new(exp)).into()
            }
            AAggExpr::Quantile { expr, quantile } => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Quantile {
                    expr: Box::new(exp),
                    quantile,
                }
                .into()
            }
            AAggExpr::Sum(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Sum(Box::new(exp)).into()
            }
            AAggExpr::Std(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Std(Box::new(exp)).into()
            }
            AAggExpr::Var(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Var(Box::new(exp)).into()
            }
            AAggExpr::AggGroups(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::AggGroups(Box::new(exp)).into()
            }
            AAggExpr::Count(expr) => {
                let exp = node_to_exp(expr, expr_arena);
                AggExpr::Count(Box::new(exp)).into()
            }
        },
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
        AExpr::Udf {
            input,
            function,
            output_type,
        } => {
            let i = node_to_exp(input, expr_arena);
            Expr::Udf {
                input: Box::new(i),
                function,
                output_type,
            }
        }
        AExpr::BinaryFunction {
            input_a,
            input_b,
            function,
            output_field,
        } => Expr::BinaryFunction {
            input_a: Box::new(node_to_exp(input_a, expr_arena)),
            input_b: Box::new(node_to_exp(input_b, expr_arena)),
            function,
            output_field,
        },
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
        AExpr::Slice {
            input,
            offset,
            length,
        } => Expr::Slice {
            input: Box::new(node_to_exp(input, expr_arena)),
            offset,
            length,
        },
        AExpr::Wildcard => Expr::Wildcard,
    }
}

pub(crate) fn node_to_lp(
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
            aggregate,
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
            aggregate: aggregate
                .into_iter()
                .map(|n| node_to_exp(n, expr_arena))
                .collect(),
            cache,
        },
        #[cfg(feature = "parquet")]
        ALogicalPlan::ParquetScan {
            path,
            schema,
            with_columns,
            predicate,
            aggregate,
            stop_after_n_rows,
            cache,
        } => LogicalPlan::ParquetScan {
            path,
            schema,
            with_columns,
            predicate: predicate.map(|n| node_to_exp(n, expr_arena)),
            aggregate: aggregate
                .into_iter()
                .map(|n| node_to_exp(n, expr_arena))
                .collect(),
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
        ALogicalPlan::Explode { input, columns } => {
            let input = Box::new(node_to_lp(input, expr_arena, lp_arena));
            LogicalPlan::Explode { input, columns }
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
            apply,
        } => {
            let i = node_to_lp(input, expr_arena, lp_arena);
            let a = aggs.iter().map(|x| node_to_exp(*x, expr_arena)).collect();
            let keys = Arc::new(keys.iter().map(|x| node_to_exp(*x, expr_arena)).collect());

            LogicalPlan::Aggregate {
                input: Box::new(i),
                keys,
                aggs: a,
                schema,
                apply,
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

            let l_on = left_on
                .into_iter()
                .map(|n| node_to_exp(n, expr_arena))
                .collect();
            let r_on = right_on
                .into_iter()
                .map(|n| node_to_exp(n, expr_arena))
                .collect();

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
        ALogicalPlan::Melt {
            input,
            id_vars,
            value_vars,
            schema,
        } => {
            let input = node_to_lp(input, expr_arena, lp_arena);
            LogicalPlan::Melt {
                input: Box::new(input),
                id_vars,
                value_vars,
                schema,
            }
        }
        ALogicalPlan::Udf {
            input,
            function,
            predicate_pd,
            projection_pd,
            schema,
        } => {
            let input = Box::new(node_to_lp(input, expr_arena, lp_arena));
            LogicalPlan::Udf {
                input,
                function,
                predicate_pd,
                projection_pd,
                schema,
            }
        }
    }
}

pub trait OptimizationRule {
    ///  Optimize (subplan) in LogicalPlan
    ///
    /// * node - node of the (sub) logicalplan root/ node
    /// * lp_arena - LogicalPlan memory arena
    /// * expr_arena - Expression memory arena
    fn optimize_plan(
        &mut self,
        _lp_arena: &mut Arena<ALogicalPlan>,
        _expr_arena: &mut Arena<AExpr>,
        _node: Node,
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

pub struct ALogicalPlanBuilder<'a> {
    root: Node,
    expr_arena: &'a mut Arena<AExpr>,
    lp_arena: &'a mut Arena<ALogicalPlan>,
}

impl<'a> ALogicalPlanBuilder<'a> {
    pub(crate) fn new(
        root: Node,
        expr_arena: &'a mut Arena<AExpr>,
        lp_arena: &'a mut Arena<ALogicalPlan>,
    ) -> Self {
        ALogicalPlanBuilder {
            root,
            expr_arena,
            lp_arena,
        }
    }

    pub fn melt(self, id_vars: Arc<Vec<String>>, value_vars: Arc<Vec<String>>) -> Self {
        let schema = det_melt_schema(&value_vars, self.schema());

        let lp = ALogicalPlan::Melt {
            input: self.root,
            id_vars,
            value_vars,
            schema,
        };
        let node = self.lp_arena.add(lp);
        ALogicalPlanBuilder::new(node, self.expr_arena, self.lp_arena)
    }

    pub fn project_local(self, exprs: Vec<Node>) -> Self {
        let input_schema = self.lp_arena.get(self.root).schema(self.lp_arena);
        let schema = aexprs_to_schema(&exprs, input_schema, Context::Other, self.expr_arena);
        if !exprs.is_empty() {
            let lp = ALogicalPlan::LocalProjection {
                expr: exprs,
                input: self.root,
                schema: Arc::new(schema),
            };
            let node = self.lp_arena.add(lp);
            ALogicalPlanBuilder::new(node, self.expr_arena, self.lp_arena)
        } else {
            self
        }
    }

    pub fn project(self, exprs: Vec<Node>) -> Self {
        let input_schema = self.lp_arena.get(self.root).schema(self.lp_arena);
        let schema = aexprs_to_schema(&exprs, input_schema, Context::Other, self.expr_arena);

        // if len == 0, no projection has to be done. This is a select all operation.
        if !exprs.is_empty() {
            let lp = ALogicalPlan::Projection {
                expr: exprs,
                input: self.root,
                schema: Arc::new(schema),
            };
            let node = self.lp_arena.add(lp);
            ALogicalPlanBuilder::new(node, self.expr_arena, self.lp_arena)
        } else {
            self
        }
    }
    pub fn into_node(self) -> Node {
        self.root
    }

    pub fn build(self) -> ALogicalPlan {
        self.lp_arena.take(self.root)
    }

    pub(crate) fn schema(&self) -> &Schema {
        self.lp_arena.get(self.root).schema(self.lp_arena)
    }

    pub(crate) fn with_columns(self, exprs: Vec<Node>) -> Self {
        // current schema
        let schema = self.schema();

        let mut new_fields = schema.fields().clone();

        for e in &exprs {
            let field = self
                .expr_arena
                .get(*e)
                .to_field(schema, Context::Other, self.expr_arena)
                .unwrap();
            match schema.index_of(field.name()) {
                Ok(idx) => {
                    new_fields[idx] = field;
                }
                Err(_) => new_fields.push(field),
            }
        }

        let new_schema = Schema::new(new_fields);

        let lp = ALogicalPlan::HStack {
            input: self.root,
            exprs,
            schema: Arc::new(new_schema),
        };
        let root = self.lp_arena.add(lp);
        Self::new(root, self.expr_arena, self.lp_arena)
    }

    pub fn groupby(
        self,
        keys: Vec<Node>,
        aggs: Vec<Node>,
        apply: Option<Arc<dyn DataFrameUdf>>,
    ) -> Self {
        debug_assert!(!keys.is_empty());
        let current_schema = self.schema();
        // TODO! add this line if LogicalPlan is dropped in favor of ALogicalPlan
        // let aggs = rewrite_projections(aggs, current_schema);

        let schema1 = aexprs_to_schema(&keys, current_schema, Context::Other, self.expr_arena);
        let schema2 =
            aexprs_to_schema(&aggs, current_schema, Context::Aggregation, self.expr_arena);

        let schema = Schema::try_merge(&[schema1, schema2]).unwrap();

        let lp = ALogicalPlan::Aggregate {
            input: self.root,
            keys,
            aggs,
            schema: Arc::new(schema),
            apply,
        };
        let root = self.lp_arena.add(lp);
        Self::new(root, self.expr_arena, self.lp_arena)
    }

    pub fn join(
        self,
        other: Node,
        how: JoinType,
        left_on: Vec<Node>,
        right_on: Vec<Node>,
        allow_par: bool,
        force_par: bool,
    ) -> Self {
        let schema_left = self.schema();
        let schema_right = self.lp_arena.get(other).schema(self.lp_arena);

        // column names of left table
        let mut names: HashSet<&String, RandomState> = HashSet::with_capacity_and_hasher(
            schema_left.len() + schema_right.len(),
            Default::default(),
        );
        // fields of new schema
        let mut fields = Vec::with_capacity(schema_left.len() + schema_right.len());

        for f in schema_left.fields() {
            names.insert(f.name());
            fields.push(f.clone());
        }

        let right_names: HashSet<_, RandomState> = right_on
            .iter()
            .map(|e| match self.expr_arena.get(*e) {
                AExpr::Alias(_, name) => name.clone(),
                AExpr::Column(name) => name.clone(),
                _ => panic!("could not determine join column names"),
            })
            .collect();

        for f in schema_right.fields() {
            let name = f.name();
            if !right_names.contains(name) {
                if names.contains(name) {
                    let new_name = format!("{}_right", name);
                    let field = Field::new(&new_name, f.data_type().clone());
                    fields.push(field)
                } else {
                    fields.push(f.clone())
                }
            }
        }

        let schema = Arc::new(Schema::new(fields));

        let lp = ALogicalPlan::Join {
            input_left: self.root,
            input_right: other,
            how,
            schema,
            left_on,
            right_on,
            allow_par,
            force_par,
        };
        let root = self.lp_arena.add(lp);
        Self::new(root, self.expr_arena, self.lp_arena)
    }
}
