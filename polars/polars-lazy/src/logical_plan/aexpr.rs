use crate::logical_plan::Context;
use crate::prelude::*;
use polars_arrow::prelude::QuantileInterpolOptions;
use polars_core::prelude::*;
use polars_core::utils::{get_supertype, get_time_units};
use polars_utils::arena::{Arena, Node};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum AAggExpr {
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
        interpol: QuantileInterpolOptions,
    },
    Sum(Node),
    Count(Node),
    Std(Node),
    Var(Node),
    AggGroups(Node),
}

// AExpr representation of Nodes which are allocated in an Arena
#[derive(Clone, Debug)]
pub enum AExpr {
    IsUnique(Node),
    Duplicated(Node),
    Reverse(Node),
    Explode(Node),
    Alias(Node, Arc<str>),
    Column(Arc<str>),
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
    Function {
        input: Vec<Node>,
        function: NoEq<Arc<dyn SeriesUdf>>,
        output_type: GetOutput,
        options: FunctionOptions,
    },
    Shift {
        input: Node,
        periods: i64,
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
            Count => Ok(Field::new("count", DataType::UInt32)),
            Window { function, .. } => {
                let e = arena.get(*function);
                e.to_field(schema, ctxt, arena)
            }
            IsUnique(expr) => {
                let field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                Ok(Field::new(field.name(), DataType::Boolean))
            }
            Duplicated(expr) => {
                let field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                Ok(Field::new(field.name(), DataType::Boolean))
            }
            Reverse(expr) => arena.get(*expr).to_field(schema, ctxt, arena),
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
            Column(name) => schema
                .get_field(name)
                .ok_or_else(|| PolarsError::NotFound(name.to_string())),
            Literal(sv) => Ok(Field::new("literal", sv.get_datatype())),
            BinaryExpr { left, right, op } => {
                use DataType::*;

                let left_type = arena.get(*left).get_type(schema, ctxt, arena)?;
                let right_type = arena.get(*right).get_type(schema, ctxt, arena)?;

                let expr_type = match op {
                    Operator::Lt
                    | Operator::Gt
                    | Operator::Eq
                    | Operator::NotEq
                    | Operator::And
                    | Operator::LtEq
                    | Operator::GtEq
                    | Operator::Or => DataType::Boolean,
                    Operator::Minus => match (left_type, right_type) {
                        // T - T != T if T is a datetime / date
                        (Datetime(tul, _), Datetime(tur, _)) => {
                            Duration(get_time_units(&tul, &tur))
                        }
                        (Date, Date) => Duration(TimeUnit::Milliseconds),
                        (left, right) => get_supertype(&left, &right)?,
                    },
                    _ => get_supertype(&left_type, &right_type)?,
                };

                let out_field;
                let out_name = {
                    out_field = arena.get(*left).to_field(schema, ctxt, arena)?;
                    out_field.name().as_str()
                };

                Ok(Field::new(out_name, expr_type))
            }
            Not(_) => Ok(Field::new("not", DataType::Boolean)),
            IsNull(_) => Ok(Field::new("is_null", DataType::Boolean)),
            IsNotNull(_) => Ok(Field::new("is_not_null", DataType::Boolean)),
            Sort { expr, .. } => arena.get(*expr).to_field(schema, ctxt, arena),
            Take { expr, .. } => arena.get(*expr).to_field(schema, ctxt, arena),
            SortBy { expr, .. } => arena.get(*expr).to_field(schema, ctxt, arena),
            Filter { input, .. } => arena.get(*input).to_field(schema, ctxt, arena),
            Agg(agg) => {
                use AAggExpr::*;
                match agg {
                    Max(expr) | Sum(expr) | Min(expr) | First(expr) | Last(expr) => {
                        arena.get(*expr).to_field(schema, ctxt, arena)
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
                        field.coerce(DataType::Float64);
                        Ok(field)
                    }
                    List(expr) => {
                        let mut field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        field.coerce(DataType::List(field.data_type().clone().into()));
                        Ok(field)
                    }
                    Std(expr) => {
                        let mut field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        field.coerce(DataType::Float64);
                        Ok(field)
                    }
                    Var(expr) => {
                        let mut field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        field.coerce(DataType::Float64);
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
                        field.coerce(DataType::Float64);
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
                    let st = get_supertype(truthy.data_type(), falsy.data_type())?;
                    truthy.coerce(st);
                    Ok(truthy)
                }
            }
            Function {
                output_type, input, ..
            } => {
                let fields = input
                    .iter()
                    .map(|node| arena.get(*node).to_field(schema, ctxt, arena))
                    .collect::<Result<Vec<_>>>()?;
                Ok(output_type.get_field(schema, ctxt, &fields))
            }
            Shift { input, .. } => arena.get(*input).to_field(schema, ctxt, arena),
            Slice { input, .. } => arena.get(*input).to_field(schema, ctxt, arena),
            Wildcard => panic!("should be no wildcard at this point"),
            Nth(_) => panic!("should be no nth at this point"),
        }
    }
}
