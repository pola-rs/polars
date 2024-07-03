use super::*;
use crate::plans::conversion::rewrite_projections;

// Except for Opaque functions, this only has the DSL name of the function.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DslFunction {
    FunctionNode(FunctionNode),
    Explode {
        columns: Vec<Expr>,
    },
    Unpivot {
        args: UnpivotArgs,
    },
    RowIndex {
        name: Arc<str>,
        offset: Option<IdxSize>,
    },
    Rename {
        existing: Arc<[SmartString]>,
        new: Arc<[SmartString]>,
    },
    Stats(StatsFunction),
    /// FillValue
    FillNan(Expr),
    Drop(DropFunction),
}

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DropFunction {
    /// Columns that are going to be dropped
    pub(crate) to_drop: PlHashSet<String>,
    /// If `true`, performs a check for each item in `to_drop` against the schema. Returns an
    /// `ColumnNotFound` error if the column does not exist in the schema.
    pub(crate) strict: bool,
}

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StatsFunction {
    Var {
        ddof: u8,
    },
    Std {
        ddof: u8,
    },
    Quantile {
        quantile: Expr,
        interpol: QuantileInterpolOptions,
    },
    Median,
    Mean,
    Sum,
    Min,
    Max,
}

impl DslFunction {
    pub(crate) fn into_function_node(self, input_schema: &Schema) -> PolarsResult<FunctionNode> {
        let function = match self {
            DslFunction::Explode { columns } => {
                let columns = rewrite_projections(columns, input_schema, &[])?;
                // columns to string
                let columns = columns
                    .iter()
                    .map(|e| {
                        let Expr::Column(name) = e else {
                            polars_bail!(InvalidOperation: "expected column expression")
                        };

                        polars_ensure!(input_schema.contains(name), ColumnNotFound: "{name}");

                        Ok(name.clone())
                    })
                    .collect::<PolarsResult<Arc<[Arc<str>]>>>()?;
                FunctionNode::Explode {
                    columns,
                    schema: Default::default(),
                }
            },
            DslFunction::Unpivot { args } => FunctionNode::Unpivot {
                args: Arc::new(args),
                schema: Default::default(),
            },
            DslFunction::FunctionNode(func) => func,
            DslFunction::RowIndex { name, offset } => FunctionNode::RowIndex {
                name,
                offset,
                schema: Default::default(),
            },
            DslFunction::Rename { existing, new } => {
                let swapping = new.iter().any(|name| input_schema.get(name).is_some());

                // Check if the name exists.
                for name in existing.iter() {
                    let _ = input_schema.try_get(name)?;
                }

                FunctionNode::Rename {
                    existing,
                    new,
                    swapping,
                    schema: Default::default(),
                }
            },
            DslFunction::Stats(_) | DslFunction::FillNan(_) | DslFunction::Drop(_) => {
                // We should not reach this.
                panic!("impl error")
            },
        };
        Ok(function)
    }
}

impl Debug for DslFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl Display for DslFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use DslFunction::*;
        match self {
            FunctionNode(inner) => write!(f, "{inner}"),
            Explode { .. } => write!(f, "EXPLODE"),
            Unpivot { .. } => write!(f, "UNPIVOT"),
            RowIndex { .. } => write!(f, "WITH ROW INDEX"),
            Stats(_) => write!(f, "STATS"),
            FillNan(_) => write!(f, "FILL NAN"),
            Drop(_) => write!(f, "DROP"),
            Rename { .. } => write!(f, "RENAME"),
        }
    }
}

impl From<FunctionNode> for DslFunction {
    fn from(value: FunctionNode) -> Self {
        DslFunction::FunctionNode(value)
    }
}
