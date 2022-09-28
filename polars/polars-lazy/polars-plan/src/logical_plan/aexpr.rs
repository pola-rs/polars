use std::sync::Arc;

use polars_arrow::prelude::QuantileInterpolOptions;
use polars_core::prelude::*;
use polars_core::utils::{get_time_units, try_get_supertype};
use polars_utils::arena::{Arena, Node};

use crate::dsl::function_expr::FunctionExpr;
use crate::logical_plan::Context;
use crate::prelude::names::COUNT;
use crate::prelude::*;

#[derive(Clone, Debug)]
pub enum AAggExpr {
    Min {
        input: Node,
        propagate_nans: bool,
    },
    Max {
        input: Node,
        propagate_nans: bool,
    },
    Median(Node),
    NUnique(Node),
    First(Node),
    Last(Node),
    Mean(Node),
    List(Node),
    Quantile {
        expr: Node,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    },
    Sum(Node),
    Count(Node),
    Std(Node, u8),
    Var(Node, u8),
    AggGroups(Node),
}

// AExpr representation of Nodes which are allocated in an Arena
#[derive(Clone, Debug)]
pub enum AExpr {
    Explode(Node),
    Alias(Node, Arc<str>),
    Column(Arc<str>),
    Literal(LiteralValue),
    BinaryExpr {
        left: Node,
        op: Operator,
        right: Node,
    },
    Cast {
        expr: Node,
        data_type: DataType,
        strict: bool,
    },
    Sort {
        expr: Node,
        options: SortOptions,
    },
    Take {
        expr: Node,
        idx: Node,
    },
    SortBy {
        expr: Node,
        by: Vec<Node>,
        reverse: Vec<bool>,
    },
    Filter {
        input: Node,
        by: Node,
    },
    Agg(AAggExpr),
    Ternary {
        predicate: Node,
        truthy: Node,
        falsy: Node,
    },
    AnonymousFunction {
        input: Vec<Node>,
        function: SpecialEq<Arc<dyn SeriesUdf>>,
        output_type: GetOutput,
        options: FunctionOptions,
    },
    Function {
        /// function arguments
        input: Vec<Node>,
        /// function to apply
        function: FunctionExpr,
        options: FunctionOptions,
    },
    Window {
        function: Node,
        partition_by: Vec<Node>,
        order_by: Option<Node>,
        options: WindowOptions,
    },
    Wildcard,
    Slice {
        input: Node,
        offset: Node,
        length: Node,
    },
    Count,
    Nth(i64),
}

impl Default for AExpr {
    fn default() -> Self {
        AExpr::Wildcard
    }
}

impl AExpr {
    /// Any expression that is sensitive to the number of elements in a group
    /// - Aggregations
    /// - Sorts
    /// - Counts
    /// - ..
    pub(crate) fn groups_sensitive(&self) -> bool {
        use AExpr::*;
        match self {
            Function { options, .. } | AnonymousFunction { options, .. } => {
                options.collect_groups == ApplyOptions::ApplyGroups
            }
            Sort { .. }
            | SortBy { .. }
            | Agg { .. }
            | Window { .. }
            | Count
            | Slice { .. }
            | Take { .. }
            | Nth(_)
             => true,
            | Alias(_, _)
            | Explode(_)
            | Column(_)
            | Literal(_)
            // a caller should traverse binary and ternary
            // to determine if the whole expr. is group sensitive
            | BinaryExpr { .. }
            | Ternary { .. }
            | Wildcard
            | Cast { .. }
            | Filter { .. } => false,
        }
    }

    /// This should be a 1 on 1 copy of the get_type method of Expr until Expr is completely phased out.
    pub(crate) fn get_type(
        &self,
        schema: &Schema,
        ctxt: Context,
        arena: &Arena<AExpr>,
    ) -> PolarsResult<DataType> {
        self.to_field(schema, ctxt, arena)
            .map(|f| f.data_type().clone())
    }

    pub(crate) fn replace_input(self, input: Node) -> Self {
        use AExpr::*;
        match self {
            Alias(_, name) => Alias(input, name),
            Cast {
                expr: _,
                data_type,
                strict,
            } => Cast {
                expr: input,
                data_type,
                strict,
            },
            _ => todo!(),
        }
    }

    pub(crate) fn get_input(&self) -> Node {
        use AExpr::*;
        match self {
            Alias(input, _) => *input,
            Cast { expr, .. } => *expr,
            _ => todo!(),
        }
    }

    /// Get Field result of the expression. The schema is the input data.
    pub(crate) fn to_field(
        &self,
        schema: &Schema,
        ctxt: Context,
        arena: &Arena<AExpr>,
    ) -> PolarsResult<Field> {
        use AExpr::*;
        match self {
            Count => Ok(Field::new(COUNT, DataType::UInt32)),
            Window { function, .. } => {
                let e = arena.get(*function);
                e.to_field(schema, ctxt, arena)
            }
            Explode(expr) => {
                let field = arena.get(*expr).to_field(schema, ctxt, arena)?;

                if let DataType::List(inner) = field.data_type() {
                    Ok(Field::new(field.name(), *inner.clone()))
                } else {
                    Ok(field)
                }
            }
            Alias(expr, name) => Ok(Field::new(
                name,
                arena.get(*expr).get_type(schema, ctxt, arena)?,
            )),
            Column(name) => {
                let field = schema
                    .get_field(name)
                    .ok_or_else(|| PolarsError::NotFound(name.to_string().into()));

                match ctxt {
                    Context::Default => field,
                    Context::Aggregation => field.map(|mut field| {
                        let dtype = DataType::List(Box::new(field.data_type().clone()));
                        field.coerce(dtype);
                        field
                    }),
                }
            }
            Literal(sv) => Ok(Field::new("literal", sv.get_datatype())),
            BinaryExpr { left, right, op } => {
                use DataType::*;

                let field = match op {
                    Operator::Lt
                    | Operator::Gt
                    | Operator::Eq
                    | Operator::NotEq
                    | Operator::And
                    | Operator::LtEq
                    | Operator::GtEq
                    | Operator::Or => {
                        let out_field;
                        let out_name = {
                            out_field = arena.get(*left).to_field(schema, ctxt, arena)?;
                            out_field.name().as_str()
                        };
                        Field::new(out_name, DataType::Boolean)
                    }
                    _ => {
                        // don't traverse tree until strictly needed. Can have terrible performance.
                        // # 3210

                        // take the left field as a whole.
                        // don't take dtype and name separate as that splits the tree every node
                        // leading to quadratic behavior. # 4736
                        let mut left_field = arena.get(*left).to_field(schema, ctxt, arena)?;
                        let right_type = arena.get(*right).get_type(schema, ctxt, arena)?;

                        let super_type = match op {
                            Operator::Minus => match (&left_field.dtype, right_type) {
                                // T - T != T if T is a datetime / date
                                (Datetime(tul, _), Datetime(tur, _)) => {
                                    Duration(get_time_units(tul, &tur))
                                }
                                (Date, Date) => Duration(TimeUnit::Milliseconds),
                                (left, right) => try_get_supertype(left, &right)?,
                            },
                            _ => try_get_supertype(&left_field.dtype, &right_type)?,
                        };
                        left_field.coerce(super_type);
                        left_field
                    }
                };

                Ok(field)
            }
            Sort { expr, .. } => arena.get(*expr).to_field(schema, ctxt, arena),
            Take { expr, .. } => arena.get(*expr).to_field(schema, ctxt, arena),
            SortBy { expr, .. } => arena.get(*expr).to_field(schema, ctxt, arena),
            Filter { input, .. } => arena.get(*input).to_field(schema, ctxt, arena),
            Agg(agg) => {
                use AAggExpr::*;
                match agg {
                    Max { input: expr, .. }
                    | Sum(expr)
                    | Min { input: expr, .. }
                    | First(expr)
                    | Last(expr) => {
                        // default context because `col()` would return a list in aggregation context
                        arena.get(*expr).to_field(schema, Context::Default, arena)
                    }
                    Median(expr) => {
                        let mut field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        if field.data_type() != &DataType::Utf8 {
                            field.coerce(DataType::Float64);
                        }
                        Ok(field)
                    }
                    Mean(expr) => {
                        let mut field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        coerce_numeric_aggregation(&mut field);
                        Ok(field)
                    }
                    List(expr) => {
                        // default context because `col()` would return a list in aggregation context
                        let mut field =
                            arena.get(*expr).to_field(schema, Context::Default, arena)?;
                        field.coerce(DataType::List(field.data_type().clone().into()));
                        Ok(field)
                    }
                    Std(expr, _) => {
                        let mut field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        coerce_numeric_aggregation(&mut field);
                        Ok(field)
                    }
                    Var(expr, _) => {
                        let mut field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        coerce_numeric_aggregation(&mut field);
                        Ok(field)
                    }
                    NUnique(expr) => {
                        let mut field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        field.coerce(DataType::UInt32);
                        Ok(field)
                    }
                    Count(expr) => {
                        let mut field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        field.coerce(DataType::UInt32);
                        Ok(field)
                    }
                    AggGroups(expr) => {
                        let mut field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        field.coerce(DataType::List(DataType::UInt32.into()));
                        Ok(field)
                    }
                    Quantile { expr, .. } => {
                        let mut field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        coerce_numeric_aggregation(&mut field);
                        Ok(field)
                    }
                }
            }
            Cast {
                expr, data_type, ..
            } => {
                let field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                Ok(Field::new(field.name(), data_type.clone()))
            }
            Ternary { truthy, falsy, .. } => {
                let mut truthy = arena.get(*truthy).to_field(schema, ctxt, arena)?;
                let falsy = arena.get(*falsy).to_field(schema, ctxt, arena)?;
                if let DataType::Null = *truthy.data_type() {
                    truthy.coerce(falsy.data_type().clone());
                    Ok(truthy)
                } else {
                    let st = try_get_supertype(truthy.data_type(), falsy.data_type())?;
                    truthy.coerce(st);
                    Ok(truthy)
                }
            }
            AnonymousFunction {
                output_type, input, ..
            } => {
                let fields = input
                    .iter()
                    // default context because `col()` would return a list in aggregation context
                    .map(|node| arena.get(*node).to_field(schema, Context::Default, arena))
                    .collect::<PolarsResult<Vec<_>>>()?;
                Ok(output_type.get_field(schema, ctxt, &fields))
            }
            Function {
                function, input, ..
            } => {
                let fields = input
                    .iter()
                    // default context because `col()` would return a list in aggregation context
                    .map(|node| arena.get(*node).to_field(schema, Context::Default, arena))
                    .collect::<PolarsResult<Vec<_>>>()?;
                function.get_field(schema, ctxt, &fields)
            }
            Slice { input, .. } => arena.get(*input).to_field(schema, ctxt, arena),
            Wildcard => panic!("should be no wildcard at this point"),
            Nth(_) => panic!("should be no nth at this point"),
        }
    }
}

fn coerce_numeric_aggregation(field: &mut Field) {
    match field.dtype {
        DataType::Duration(_) => {
            // pass
        }
        DataType::Float32 => {
            // pass
        }
        _ => {
            field.coerce(DataType::Float64);
        }
    }
}
