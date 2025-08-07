use polars_compute::rolling::QuantileMethod;
use strum_macros::IntoStaticStr;

use super::*;

#[cfg(feature = "python")]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub enum DslFunction {
    RowIndex {
        name: PlSmallStr,
        offset: Option<IdxSize>,
    },
    // This is both in DSL and IR because we want to be able to serialize it.
    #[cfg(feature = "python")]
    OpaquePython(OpaquePythonUdf),
    Explode {
        columns: Selector,
        allow_empty: bool,
    },
    #[cfg(feature = "pivot")]
    Unpivot {
        args: UnpivotArgsDSL,
    },
    Rename {
        existing: Arc<[PlSmallStr]>,
        new: Arc<[PlSmallStr]>,
        strict: bool,
    },
    Unnest(Selector),
    Stats(StatsFunction),
    /// FillValue
    FillNan(Expr),
    // Function that is already converted to IR.
    #[cfg_attr(any(feature = "serde", feature = "dsl-schema"), serde(skip))]
    FunctionIR(FunctionIR),
}

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum StatsFunction {
    Var {
        ddof: u8,
    },
    Std {
        ddof: u8,
    },
    Quantile {
        quantile: Expr,
        method: QuantileMethod,
    },
    Median,
    Mean,
    Sum,
    Min,
    Max,
}

pub(crate) fn validate_columns_in_input<S: AsRef<str>, I: IntoIterator<Item = S>>(
    columns: I,
    input_schema: &Schema,
    operation_name: &str,
) -> PolarsResult<()> {
    let columns = columns.into_iter();
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
                let on = args.on.into_columns(input_schema, &Default::default())?;
                let index = args.index.into_columns(input_schema, &Default::default())?;

                let args = UnpivotArgsIR {
                    on: on.into_iter().collect(),
                    index: index.into_iter().collect(),
                    variable_name: args.variable_name.clone(),
                    value_name: args.value_name,
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
            DslFunction::Unnest(selector) => {
                let columns = selector.into_columns(input_schema, &Default::default())?;
                let columns = columns.into_iter().collect();
                FunctionIR::Unnest { columns }
            },
            #[cfg(feature = "python")]
            DslFunction::OpaquePython(inner) => FunctionIR::OpaquePython(inner),
            DslFunction::Stats(_)
            | DslFunction::FillNan(_)
            | DslFunction::Rename { .. }
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
