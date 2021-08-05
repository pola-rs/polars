use crate::logical_plan::Context;
use crate::prelude::*;
use crate::utils::rename_field;
use polars_core::frame::groupby::{fmt_groupby_column, GroupByMethod};
use polars_core::prelude::*;
use polars_core::utils::{get_supertype, Arena, Node};
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
    Quantile { expr: Node, quantile: f64 },
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
    Take {
        expr: Node,
        idx: Node,
    },
    SortBy {
        expr: Node,
        by: Node,
        reverse: bool,
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
        output_type: Option<DataType>,
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
    },
    Wildcard,
    Slice {
        input: Node,
        offset: i64,
        length: usize,
    },
    BinaryFunction {
        input_a: Node,
        input_b: Node,
        function: NoEq<Arc<dyn SeriesBinaryUdf>>,
        /// Delays output type evaluation until input schema is known.
        output_field: NoEq<Arc<dyn BinaryUdfOutputField>>,
    },
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
            Window { function, .. } => arena.get(*function).to_field(schema, ctxt, arena),
            IsUnique(expr) => {
                let field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                Ok(Field::new(field.name(), DataType::Boolean))
            }
            Duplicated(expr) => {
                let field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                Ok(Field::new(field.name(), DataType::Boolean))
            }
            Reverse(expr) => arena.get(*expr).to_field(schema, ctxt, arena),
            Explode(expr) => arena.get(*expr).to_field(schema, ctxt, arena),
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
                    Operator::Lt
                    | Operator::Gt
                    | Operator::Eq
                    | Operator::NotEq
                    | Operator::And
                    | Operator::LtEq
                    | Operator::GtEq
                    | Operator::Or => DataType::Boolean,
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
            Take { expr, .. } => arena.get(*expr).to_field(schema, ctxt, arena),
            SortBy { expr, .. } => arena.get(*expr).to_field(schema, ctxt, arena),
            Filter { input, .. } => arena.get(*input).to_field(schema, ctxt, arena),
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
                    Mean(expr) => {
                        let mut field = field_by_context(
                            arena.get(*expr).to_field(schema, ctxt, arena)?,
                            ctxt,
                            GroupByMethod::Mean,
                        );
                        field.coerce(DataType::Float64);
                        field
                    }
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
                        let mut field = field_by_context(field, ctxt, GroupByMethod::Std);
                        field.coerce(DataType::Float64);
                        field
                    }
                    Var(expr) => {
                        let field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        let field = Field::new(field.name(), DataType::Float64);
                        let mut field = field_by_context(field, ctxt, GroupByMethod::Var);
                        field.coerce(DataType::Float64);
                        field
                    }
                    NUnique(expr) => {
                        let field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                        let field = Field::new(field.name(), DataType::UInt32);
                        match ctxt {
                            Context::Default => field,
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
                            Context::Default => field,
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
                    Quantile { expr, quantile } => {
                        let mut field = field_by_context(
                            arena.get(*expr).to_field(schema, ctxt, arena)?,
                            ctxt,
                            GroupByMethod::Quantile(*quantile),
                        );
                        field.coerce(DataType::Float64);
                        field
                    }
                };
                Ok(field)
            }
            Cast { expr, data_type } => {
                let field = arena.get(*expr).to_field(schema, ctxt, arena)?;
                Ok(Field::new(field.name(), data_type.clone()))
            }
            Ternary { truthy, .. } => arena.get(*truthy).to_field(schema, ctxt, arena),
            Function {
                output_type, input, ..
            } => match output_type {
                None => arena.get(input[0]).to_field(schema, ctxt, arena),
                Some(output_type) => {
                    let input_field = arena.get(input[0]).to_field(schema, ctxt, arena)?;
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
                // TODO: remove Option?
                Ok(out.expect("field should be set"))
            }
            Shift { input, .. } => arena.get(*input).to_field(schema, ctxt, arena),
            Slice { input, .. } => arena.get(*input).to_field(schema, ctxt, arena),
            Wildcard => panic!("should be no wildcard at this point"),
        }
    }

    /// Check if AExpr equality. The nodes may differ.
    ///
    /// For instance: there can be two columns "foo" in the memory arena. These are equal,
    /// but would have different node values.
    #[cfg(feature = "private")]
    pub(crate) fn eq(node_left: Node, node_right: Node, expr_arena: &Arena<AExpr>) -> bool {
        use crate::logical_plan::iterator::ArenaExprIter;
        let cmp = |(node_left, node_right)| {
            use AExpr::*;
            match (expr_arena.get(node_left), expr_arena.get(node_right)) {
                (Alias(_, name_l), Alias(_, name_r)) => name_l == name_r,
                (Column(name_l), Column(name_r)) => name_l == name_r,
                (Literal(left), Literal(right)) => left == right,
                (BinaryExpr { op: l, .. }, BinaryExpr { op: r, .. }) => l == r,
                (Cast { data_type: l, .. }, Cast { data_type: r, .. }) => l == r,
                (Sort { reverse: l, .. }, Sort { reverse: r, .. }) => l == r,
                (SortBy { reverse: l, .. }, SortBy { reverse: r, .. }) => l == r,
                (Shift { periods: l, .. }, Shift { periods: r, .. }) => l == r,
                (
                    Slice {
                        offset: offset_l,
                        length: length_l,
                        ..
                    },
                    Slice {
                        offset: offset_r,
                        length: length_r,
                        ..
                    },
                ) => offset_l == offset_r && length_l == length_r,
                (a, b) => std::mem::discriminant(a) == std::mem::discriminant(b),
            }
        };

        expr_arena
            .iter(node_left)
            .zip(expr_arena.iter(node_right))
            .map(|(tpll, tplr)| (tpll.0, tplr.0))
            .all(cmp)
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
        Context::Default => field,
        Context::Aggregation => {
            let new_name = fmt_groupby_column(field.name(), groupby_method);
            rename_field(&field, &new_name)
        }
    }
}
