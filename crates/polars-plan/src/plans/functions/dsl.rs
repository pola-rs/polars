use strum_macros::IntoStaticStr;

use super::*;

#[cfg(feature = "python")]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct OpaquePythonUdf {
    pub function: PythonFunction,
    pub schema: Option<SchemaRef>,
    ///  allow predicate pushdown optimizations
    pub predicate_pd: bool,
    ///  allow projection pushdown optimizations
    pub projection_pd: bool,
    pub streamable: bool,
    pub validate_output: bool,
}

// Except for Opaque functions, this only has the DSL name of the function.
#[derive(Clone, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub enum DslFunction {
    // Function that is already converted to IR.
    #[cfg_attr(feature = "serde", serde(skip))]
    FunctionIR(FunctionIR),
    // This is both in DSL and IR because we want to be able to serialize it.
    #[cfg(feature = "python")]
    OpaquePython(OpaquePythonUdf),
    Explode {
        columns: Vec<Selector>,
        allow_empty: bool,
    },
    #[cfg(feature = "pivot")]
    Unpivot {
        args: UnpivotArgsDSL,
    },
    RowIndex {
        name: Arc<str>,
        offset: Option<IdxSize>,
    },
    Rename {
        existing: Arc<[SmartString]>,
        new: Arc<[SmartString]>,
    },
    Unnest(Vec<Selector>),
    Stats(StatsFunction),
    /// FillValue
    FillNan(Expr),
    Drop(DropFunction),
}

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DropFunction {
    /// Columns that are going to be dropped
    pub(crate) to_drop: Vec<Selector>,
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

pub(crate) fn validate_columns_in_input<S: AsRef<str>>(
    columns: &[S],
    input_schema: &Schema,
    operation_name: &str,
) -> PolarsResult<()> {
    for c in columns {
        polars_ensure!(input_schema.contains(c.as_ref()), ColumnNotFound: "'{}' on column: '{}' is invalid\n\nSchema at this point: {:?}", operation_name, c.as_ref(), input_schema)
    }
    Ok(())
}

impl DslFunction {
    pub(crate) fn into_function_ir(self, input_schema: &Schema) -> PolarsResult<FunctionIR> {
        let function = match self {
            #[cfg(feature = "pivot")]
            DslFunction::Unpivot { args } => {
                let on = expand_selectors(args.on, input_schema, &[])?;
                let index = expand_selectors(args.index, input_schema, &[])?;
                validate_columns_in_input(on.as_ref(), input_schema, "unpivot")?;
                validate_columns_in_input(index.as_ref(), input_schema, "unpivot")?;

                let args = UnpivotArgsIR {
                    on: on.iter().map(|s| s.as_ref().into()).collect(),
                    index: index.iter().map(|s| s.as_ref().into()).collect(),
                    variable_name: args.variable_name.map(|s| s.as_ref().into()),
                    value_name: args.value_name.map(|s| s.as_ref().into()),
                };

                FunctionIR::Unpivot {
                    args: Arc::new(args),
                    schema: Default::default(),
                }
            },
            DslFunction::FunctionIR(func) => func,
            DslFunction::RowIndex { name, offset } => FunctionIR::RowIndex {
                name,
                offset,
                schema: Default::default(),
            },
            DslFunction::Rename { existing, new } => {
                let swapping = new.iter().any(|name| input_schema.get(name).is_some());
                validate_columns_in_input(existing.as_ref(), input_schema, "rename")?;

                FunctionIR::Rename {
                    existing,
                    new,
                    swapping,
                    schema: Default::default(),
                }
            },
            DslFunction::Unnest(selectors) => {
                let columns = expand_selectors(selectors, input_schema, &[])?;
                validate_columns_in_input(columns.as_ref(), input_schema, "explode")?;
                FunctionIR::Unnest { columns }
            },
            #[cfg(feature = "python")]
            DslFunction::OpaquePython(inner) => FunctionIR::OpaquePython(inner),
            DslFunction::Stats(_)
            | DslFunction::FillNan(_)
            | DslFunction::Drop(_)
            | DslFunction::Explode { .. } => {
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
            FunctionIR(inner) => write!(f, "{inner}"),
            v => {
                let s: &str = v.into();
                write!(f, "{s}")
            },
        }
    }
}

impl From<FunctionIR> for DslFunction {
    fn from(value: FunctionIR) -> Self {
        DslFunction::FunctionIR(value)
    }
}
