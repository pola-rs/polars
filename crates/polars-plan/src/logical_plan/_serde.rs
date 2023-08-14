use serde::{Deserializer, Serializer};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UserDefinedNode {
    pub name: String,
    pub bytes: Vec<u8>,
}

pub trait FunctionRegistry {
    fn try_encode_scan(&self, _scan: &dyn AnonymousScan) -> PolarsResult<UserDefinedNode>;
    fn try_encode_udf(&self, _udf: &dyn DataFrameUdf) -> PolarsResult<UserDefinedNode>;
    fn try_decode_scan(
        &self,
        _node: &UserDefinedNode,
    ) -> PolarsResult<Option<Arc<dyn AnonymousScan>>>;
    fn try_decode_udf(
        &self,
        _node: &UserDefinedNode,
    ) -> PolarsResult<Option<Arc<dyn DataFrameUdf>>>;
}
struct DefaultFunctionRegistry;

impl FunctionRegistry for DefaultFunctionRegistry {
    fn try_encode_scan(&self, _scan: &dyn AnonymousScan) -> PolarsResult<UserDefinedNode> {
        polars_bail!(InvalidOperation: "no default implementation for encoding scans")
    }
    fn try_decode_scan(
        &self,
        _node: &UserDefinedNode,
    ) -> PolarsResult<Option<Arc<dyn AnonymousScan>>> {
        polars_bail!(InvalidOperation: "no default implementation for decoding scans")
    }

    fn try_encode_udf(&self, _udf: &dyn DataFrameUdf) -> PolarsResult<UserDefinedNode> {
        polars_bail!(InvalidOperation: "no default implementation for decoding scans")
    }

    fn try_decode_udf(
        &self,
        _node: &UserDefinedNode,
    ) -> PolarsResult<Option<Arc<dyn DataFrameUdf>>> {
        polars_bail!(InvalidOperation: "no default implementation for decoding scans")
    }
}

use super::*;

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum SerializableLogicalPlan {
    AnonymousScan {
        node: UserDefinedNode,
        file_info: FileInfo,
        predicate: Option<Expr>,
        options: Arc<AnonymousScanOptions>,
    },
    #[cfg(feature = "python")]
    PythonScan { options: PythonOptions },
    /// Filter on a boolean mask
    Selection {
        input: Box<SerializableLogicalPlan>,
        predicate: Expr,
    },
    /// Cache the input at this point in the LP
    Cache {
        input: Box<SerializableLogicalPlan>,
        id: usize,
        count: usize,
    },
    Scan {
        path: PathBuf,
        file_info: FileInfo,
        predicate: Option<Expr>,
        file_options: FileScanOptions,
        scan_type: FileScan,
    },
    // we keep track of the projection and selection as it is cheaper to first project and then filter
    /// In memory DataFrame
    DataFrameScan {
        df: Arc<DataFrame>,
        schema: SchemaRef,
        // schema of the projected file
        output_schema: Option<SchemaRef>,
        projection: Option<Arc<Vec<String>>>,
        selection: Option<Expr>,
    },
    // a projection that doesn't have to be optimized
    // or may drop projected columns if they aren't in current schema (after optimization)
    LocalProjection {
        expr: Vec<Expr>,
        input: Box<SerializableLogicalPlan>,
        schema: SchemaRef,
    },
    /// Column selection
    Projection {
        expr: Vec<Expr>,
        input: Box<SerializableLogicalPlan>,
        schema: SchemaRef,
        options: ProjectionOptions,
    },
    /// Groupby aggregation
    Aggregate {
        input: Box<SerializableLogicalPlan>,
        keys: Arc<Vec<Expr>>,
        aggs: Vec<Expr>,
        schema: SchemaRef,
        apply: Option<UserDefinedNode>,
        maintain_order: bool,
        options: Arc<GroupbyOptions>,
    },
    /// Join operation
    Join {
        input_left: Box<SerializableLogicalPlan>,
        input_right: Box<SerializableLogicalPlan>,
        schema: SchemaRef,
        left_on: Vec<Expr>,
        right_on: Vec<Expr>,
        options: Arc<JoinOptions>,
    },
    /// Adding columns to the table without a Join
    HStack {
        input: Box<SerializableLogicalPlan>,
        exprs: Vec<Expr>,
        schema: SchemaRef,
        options: ProjectionOptions,
    },
    /// Remove duplicates from the table
    Distinct {
        input: Box<SerializableLogicalPlan>,
        options: DistinctOptions,
    },
    /// Sort the table
    Sort {
        input: Box<SerializableLogicalPlan>,
        by_column: Vec<Expr>,
        args: SortArguments,
    },
    /// Slice the table
    Slice {
        input: Box<SerializableLogicalPlan>,
        offset: i64,
        len: IdxSize,
    },
    /// A (User Defined) Function
    MapFunction {
        input: Box<SerializableLogicalPlan>,
        function: FunctionNode,
    },
    Union {
        inputs: Vec<SerializableLogicalPlan>,
        options: UnionOptions,
    },
    /// Catches errors and throws them later
    #[cfg_attr(feature = "serde", serde(skip))]
    Error {
        input: Box<SerializableLogicalPlan>,
        err: ErrorStateSync,
    },
    /// This allows expressions to access other tables
    ExtContext {
        input: Box<SerializableLogicalPlan>,
        contexts: Vec<SerializableLogicalPlan>,
        schema: SchemaRef,
    },
    FileSink {
        input: Box<SerializableLogicalPlan>,
        payload: FileSinkOptions,
    },
}

impl SerializableLogicalPlan {
    fn from_logical_plan(lp: LogicalPlan, registry: &dyn FunctionRegistry) -> PolarsResult<Self> {
        match lp {
            LogicalPlan::AnonymousScan {
                function,
                file_info,
                predicate,
                options,
            } => {
                // let name = function.name();

                let node = registry.try_encode_scan(function.as_ref())?;

                Ok(SerializableLogicalPlan::AnonymousScan {
                    node,
                    file_info,
                    options,
                    predicate,
                })
            },
            #[cfg(feature = "python")]
            LogicalPlan::PythonScan { options } => {
                Ok(SerializableLogicalPlan::PythonScan { options })
            },
            LogicalPlan::Selection { input, predicate } => {
                let input = Self::from_logical_plan(*input, registry)?;
                Ok(SerializableLogicalPlan::Selection {
                    input: Box::new(input),
                    predicate,
                })
            },
            LogicalPlan::Cache { input, id, count } => {
                let input = Self::from_logical_plan(*input, registry)?;
                Ok(SerializableLogicalPlan::Cache {
                    input: Box::new(input),
                    id,
                    count,
                })
            },
            LogicalPlan::Scan {
                path,
                file_info,
                predicate,
                file_options,
                scan_type,
            } => Ok(SerializableLogicalPlan::Scan {
                path,
                file_info,
                predicate,
                file_options,
                scan_type,
            }),
            DataFrameScan {
                df,
                schema,
                output_schema,
                projection,
                selection,
            } => Ok(SerializableLogicalPlan::DataFrameScan {
                df,
                schema,
                output_schema,
                projection,
                selection,
            }),
            LogicalPlan::LocalProjection {
                expr,
                input,
                schema,
            } => {
                let input = Self::from_logical_plan(*input, registry)?;
                Ok(SerializableLogicalPlan::LocalProjection {
                    expr,
                    input: Box::new(input),
                    schema,
                })
            },
            LogicalPlan::Projection {
                expr,
                input,
                schema,
                options,
            } => {
                let input = Self::from_logical_plan(*input, registry)?;
                Ok(SerializableLogicalPlan::Projection {
                    expr,
                    input: Box::new(input),
                    schema,
                    options,
                })
            },
            LogicalPlan::Aggregate { .. } => {
                todo!()
            },
            LogicalPlan::Join {
                input_left,
                input_right,
                schema,
                left_on,
                right_on,
                options,
            } => {
                let input_left = Self::from_logical_plan(*input_left, registry)?;
                let input_right = Self::from_logical_plan(*input_right, registry)?;
                Ok(SerializableLogicalPlan::Join {
                    input_left: Box::new(input_left),
                    input_right: Box::new(input_right),
                    schema,
                    left_on,
                    right_on,
                    options,
                })
            },
            LogicalPlan::HStack {
                input,
                exprs,
                schema,
                options,
            } => {
                let input = Self::from_logical_plan(*input, registry)?;
                Ok(SerializableLogicalPlan::HStack {
                    input: Box::new(input),
                    exprs,
                    schema,
                    options,
                })
            },
            LogicalPlan::Distinct { input, options } => {
                let input = Self::from_logical_plan(*input, registry)?;
                Ok(SerializableLogicalPlan::Distinct {
                    input: Box::new(input),
                    options,
                })
            },
            LogicalPlan::Sort {
                input,
                by_column,
                args,
            } => {
                let input = Self::from_logical_plan(*input, registry)?;
                Ok(SerializableLogicalPlan::Sort {
                    input: Box::new(input),
                    by_column,
                    args,
                })
            },
            LogicalPlan::Slice { input, offset, len } => {
                let input = Self::from_logical_plan(*input, registry)?;
                Ok(SerializableLogicalPlan::Slice {
                    input: Box::new(input),
                    offset,
                    len,
                })
            },
            LogicalPlan::MapFunction { input, function } => {
                let input = Self::from_logical_plan(*input, registry)?;
                Ok(SerializableLogicalPlan::MapFunction {
                    input: Box::new(input),
                    function,
                })
            },
            LogicalPlan::Union { inputs, options } => {
                let inputs = inputs
                    .into_iter()
                    .map(|input| Self::from_logical_plan(input, registry))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(SerializableLogicalPlan::Union { inputs, options })
            },
            LogicalPlan::Error { input, err } => {
                let input = Self::from_logical_plan(*input, registry)?;
                Ok(SerializableLogicalPlan::Error {
                    input: Box::new(input),
                    err,
                })
            },
            LogicalPlan::ExtContext {
                input,
                contexts,
                schema,
            } => {
                let input = Self::from_logical_plan(*input, registry)?;
                let contexts = contexts
                    .into_iter()
                    .map(|input| Self::from_logical_plan(input, registry))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(SerializableLogicalPlan::ExtContext {
                    input: Box::new(input),
                    contexts,
                    schema,
                })
            },
            LogicalPlan::FileSink { input, payload } => {
                let input = Self::from_logical_plan(*input, registry)?;
                Ok(SerializableLogicalPlan::FileSink {
                    input: Box::new(input),
                    payload,
                })
            },
        }
    }

    fn try_into_logical_plan(self, registry: &dyn FunctionRegistry) -> PolarsResult<LogicalPlan> {
        match self {
            Self::AnonymousScan {
                node,
                file_info,
                predicate,
                options,
            } => {
                let f = registry.try_decode_scan(&node)?;
                if let Some(f) = f {
                    Ok(LogicalPlan::AnonymousScan {
                        function: f,
                        file_info,
                        options,
                        predicate,
                    })
                } else {
                    Err(PolarsError::ComputeError(
                        format!("Could not find a scan function with name: {}", &node.name).into(),
                    ))
                }
            },
            Self::Slice { input, offset, len } => {
                let input = input.try_into_logical_plan(registry)?;
                Ok(LogicalPlan::Slice {
                    input: Box::new(input),
                    offset,
                    len,
                })
            },
            #[cfg(feature = "python")]
            SerializableLogicalPlan::PythonScan { options } => {
                Ok(LogicalPlan::PythonScan { options })
            },
            SerializableLogicalPlan::Selection { input, predicate } => {
                let input = input.try_into_logical_plan(registry)?;
                Ok(LogicalPlan::Selection {
                    input: Box::new(input),
                    predicate,
                })
            },
            SerializableLogicalPlan::Cache { input, id, count } => {
                let input = input.try_into_logical_plan(registry)?;
                Ok(LogicalPlan::Cache {
                    input: Box::new(input),
                    id,
                    count,
                })
            },
            SerializableLogicalPlan::Scan {
                path,
                file_info,
                predicate,
                file_options,
                scan_type,
            } => Ok(LogicalPlan::Scan {
                path,
                file_info,
                predicate,
                file_options,
                scan_type,
            }),
            SerializableLogicalPlan::DataFrameScan {
                df,
                schema,
                output_schema,
                projection,
                selection,
            } => Ok(LogicalPlan::DataFrameScan {
                df,
                schema,
                output_schema,
                projection,
                selection,
            }),
            SerializableLogicalPlan::LocalProjection {
                expr,
                input,
                schema,
            } => {
                let input = input.try_into_logical_plan(registry)?;
                Ok(LogicalPlan::LocalProjection {
                    expr,
                    input: Box::new(input),
                    schema,
                })
            },
            SerializableLogicalPlan::Projection {
                expr,
                input,
                schema,
                options,
            } => {
                let input = input.try_into_logical_plan(registry)?;
                Ok(LogicalPlan::Projection {
                    expr,
                    input: Box::new(input),
                    schema,
                    options,
                })
            },
            SerializableLogicalPlan::Aggregate {
                apply,
                input,
                keys,
                aggs,
                schema,
                maintain_order,
                options,
            } => {
                let input = input.try_into_logical_plan(registry)?;
                let apply = apply
                    .map(|node| registry.try_decode_udf(&node))
                    .transpose()?
                    .flatten();
                Ok(LogicalPlan::Aggregate {
                    input: Box::new(input),
                    keys,
                    aggs,
                    schema,
                    apply,
                    maintain_order,
                    options,
                })
            },
            SerializableLogicalPlan::Join {
                input_left,
                input_right,
                schema,
                left_on,
                right_on,
                options,
            } => {
                let input_left = input_left.try_into_logical_plan(registry)?;
                let input_right = input_right.try_into_logical_plan(registry)?;
                Ok(LogicalPlan::Join {
                    input_left: Box::new(input_left),
                    input_right: Box::new(input_right),
                    schema,
                    left_on,
                    right_on,
                    options,
                })
            },
            SerializableLogicalPlan::HStack {
                input,
                exprs,
                schema,
                options,
            } => {
                let input = input.try_into_logical_plan(registry)?;
                Ok(LogicalPlan::HStack {
                    input: Box::new(input),
                    exprs,
                    schema,
                    options,
                })
            },
            SerializableLogicalPlan::Distinct { input, options } => {
                let input = input.try_into_logical_plan(registry)?;
                Ok(LogicalPlan::Distinct {
                    input: Box::new(input),
                    options,
                })
            },
            SerializableLogicalPlan::Sort {
                input,
                by_column,
                args,
            } => {
                let input = input.try_into_logical_plan(registry)?;
                Ok(LogicalPlan::Sort {
                    input: Box::new(input),
                    by_column,
                    args,
                })
            },

            SerializableLogicalPlan::MapFunction { input, function } => {
                let input = input.try_into_logical_plan(registry)?;
                Ok(LogicalPlan::MapFunction {
                    input: Box::new(input),
                    function,
                })
            },
            SerializableLogicalPlan::Union { inputs, options } => {
                let inputs = inputs
                    .into_iter()
                    .map(|input| input.try_into_logical_plan(registry))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(LogicalPlan::Union { inputs, options })
            },
            SerializableLogicalPlan::Error { input, err } => {
                let input = input.try_into_logical_plan(registry)?;
                Ok(LogicalPlan::Error {
                    input: Box::new(input),
                    err,
                })
            },
            SerializableLogicalPlan::ExtContext {
                input,
                contexts,
                schema,
            } => {
                let input = input.try_into_logical_plan(registry)?;
                let contexts = contexts
                    .into_iter()
                    .map(|input| input.try_into_logical_plan(registry))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(LogicalPlan::ExtContext {
                    input: Box::new(input),
                    contexts,
                    schema,
                })
            },
            SerializableLogicalPlan::FileSink { input, payload } => {
                let input = input.try_into_logical_plan(registry)?;
                Ok(LogicalPlan::FileSink {
                    input: Box::new(input),
                    payload,
                })
            },
        }
    }
}

impl LogicalPlan {
    pub fn try_serialize<S: Serializer>(
        &self,
        serializer: S,
        registry: &dyn FunctionRegistry,
    ) -> PolarsResult<S::Ok> {
        let plan = SerializableLogicalPlan::from_logical_plan(self.clone(), registry)?;
        plan.serialize(serializer).map_err(to_compute_err)
    }
    pub fn try_deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
        registry: &dyn FunctionRegistry,
    ) -> PolarsResult<Self> {
        let plan = SerializableLogicalPlan::deserialize(deserializer).map_err(to_compute_err)?;
        plan.try_into_logical_plan(registry)
    }
}
