use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use polars_arrow::error::to_compute_err;
#[cfg(feature = "parquet")]
use polars_core::cloud::CloudOptions;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserializer, Serializer};

use crate::logical_plan::LogicalPlan::DataFrameScan;
use crate::prelude::*;
use crate::utils::{expr_to_leaf_column_names, get_single_leaf};

pub(crate) mod aexpr;
pub(crate) mod alp;
pub(crate) mod anonymous_scan;

mod apply;
mod builder;
mod builder_alp;
pub mod builder_functions;
pub(crate) mod conversion;
#[cfg(feature = "debugging")]
pub(crate) mod debug;
mod file_scan;
mod format;
mod functions;
pub(crate) mod iterator;
mod lit;
pub(crate) mod optimizer;
pub(crate) mod options;
pub(crate) mod projection;
mod projection_expr;
#[cfg(feature = "python")]
mod pyarrow;
mod schema;
#[cfg(any(feature = "meta", feature = "cse"))]
pub(crate) mod tree_format;
pub mod visitor;

pub use aexpr::*;
pub use alp::*;
pub use anonymous_scan::*;
pub use apply::*;
pub use builder::*;
pub use builder_alp::*;
pub use conversion::*;
pub use file_scan::*;
pub use functions::*;
pub use iterator::*;
pub use lit::*;
pub use optimizer::*;
pub use schema::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

#[cfg(any(feature = "ipc", feature = "parquet", feature = "csv", feature = "cse"))]
pub use crate::logical_plan::optimizer::file_caching::{
    collect_fingerprints, find_column_union_and_fingerprints, FileCacher, FileFingerPrint,
};

#[derive(Clone, Copy, Debug)]
pub enum Context {
    /// Any operation that is done on groups
    Aggregation,
    /// Any operation that is done while projection/ selection of data
    Default,
}

#[derive(Debug)]
pub enum ErrorState {
    NotYetEncountered { err: PolarsError },
    AlreadyEncountered { prev_err_msg: String },
}

impl std::fmt::Display for ErrorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorState::NotYetEncountered { err } => write!(f, "NotYetEncountered({err})")?,
            ErrorState::AlreadyEncountered { prev_err_msg } => {
                write!(f, "AlreadyEncountered({prev_err_msg})")?
            },
        };

        Ok(())
    }
}

#[derive(Clone)]
pub struct ErrorStateSync(Arc<Mutex<ErrorState>>);

impl std::ops::Deref for ErrorStateSync {
    type Target = Arc<Mutex<ErrorState>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Debug for ErrorStateSync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ErrorStateSync({})", &*self.0.lock().unwrap())
    }
}

impl ErrorStateSync {
    fn take(&self) -> PolarsError {
        let mut curr_err = self.0.lock().unwrap();

        match &*curr_err {
            ErrorState::NotYetEncountered { err: polars_err } => {
                // Need to finish using `polars_err` here so that NLL considers `err` dropped
                let prev_err_msg = polars_err.to_string();
                // Place AlreadyEncountered in `self` for future users of `self`
                let prev_err = std::mem::replace(
                    &mut *curr_err,
                    ErrorState::AlreadyEncountered { prev_err_msg },
                );
                // Since we're in this branch, we know err was a NotYetEncountered
                match prev_err {
                    ErrorState::NotYetEncountered { err } => err,
                    ErrorState::AlreadyEncountered { .. } => unreachable!(),
                }
            },
            ErrorState::AlreadyEncountered { prev_err_msg } => {
                polars_err!(
                    ComputeError: "LogicalPlan already failed with error: '{}'", prev_err_msg,
                )
            },
        }
    }
}

impl From<PolarsError> for ErrorStateSync {
    fn from(err: PolarsError) -> Self {
        Self(Arc::new(Mutex::new(ErrorState::NotYetEncountered { err })))
    }
}

// https://stackoverflow.com/questions/1031076/what-are-projection-and-selection
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LogicalPlan {
    #[cfg_attr(feature = "serde", serde(skip))]
    AnonymousScan {
        function: Arc<dyn AnonymousScan>,
        file_info: FileInfo,
        predicate: Option<Expr>,
        options: Arc<AnonymousScanOptions>,
    },
    #[cfg(feature = "python")]
    PythonScan { options: PythonOptions },
    /// Filter on a boolean mask
    Selection {
        input: Box<LogicalPlan>,
        predicate: Expr,
    },
    /// Cache the input at this point in the LP
    Cache {
        input: Box<LogicalPlan>,
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
        input: Box<LogicalPlan>,
        schema: SchemaRef,
    },
    /// Column selection
    Projection {
        expr: Vec<Expr>,
        input: Box<LogicalPlan>,
        schema: SchemaRef,
        options: ProjectionOptions,
    },
    /// Groupby aggregation
    Aggregate {
        input: Box<LogicalPlan>,
        keys: Arc<Vec<Expr>>,
        aggs: Vec<Expr>,
        schema: SchemaRef,
        #[cfg_attr(feature = "serde", serde(skip))]
        apply: Option<Arc<dyn DataFrameUdf>>,
        maintain_order: bool,
        options: Arc<GroupbyOptions>,
    },
    /// Join operation
    Join {
        input_left: Box<LogicalPlan>,
        input_right: Box<LogicalPlan>,
        schema: SchemaRef,
        left_on: Vec<Expr>,
        right_on: Vec<Expr>,
        options: Arc<JoinOptions>,
    },
    /// Adding columns to the table without a Join
    HStack {
        input: Box<LogicalPlan>,
        exprs: Vec<Expr>,
        schema: SchemaRef,
        options: ProjectionOptions,
    },
    /// Remove duplicates from the table
    Distinct {
        input: Box<LogicalPlan>,
        options: DistinctOptions,
    },
    /// Sort the table
    Sort {
        input: Box<LogicalPlan>,
        by_column: Vec<Expr>,
        args: SortArguments,
    },
    /// Slice the table
    Slice {
        input: Box<LogicalPlan>,
        offset: i64,
        len: IdxSize,
    },
    /// A (User Defined) Function
    MapFunction {
        input: Box<LogicalPlan>,
        function: FunctionNode,
    },
    Union {
        inputs: Vec<LogicalPlan>,
        options: UnionOptions,
    },
    /// Catches errors and throws them later
    #[cfg_attr(feature = "serde", serde(skip))]
    Error {
        input: Box<LogicalPlan>,
        err: ErrorStateSync,
    },
    /// This allows expressions to access other tables
    ExtContext {
        input: Box<LogicalPlan>,
        contexts: Vec<LogicalPlan>,
        schema: SchemaRef,
    },
    FileSink {
        input: Box<LogicalPlan>,
        payload: FileSinkOptions,
    },
}

impl Default for LogicalPlan {
    fn default() -> Self {
        let df = DataFrame::new::<Series>(vec![]).unwrap();
        let schema = df.schema();
        DataFrameScan {
            df: Arc::new(df),
            schema: Arc::new(schema),
            output_schema: None,
            projection: None,
            selection: None,
        }
    }
}

impl LogicalPlan {
    pub fn describe(&self) -> String {
        format!("{self:#?}")
    }

    pub fn to_alp(self) -> PolarsResult<(Node, Arena<ALogicalPlan>, Arena<AExpr>)> {
        let mut lp_arena = Arena::with_capacity(16);
        let mut expr_arena = Arena::with_capacity(16);

        let node = to_alp(self, &mut expr_arena, &mut lp_arena)?;

        Ok((node, lp_arena, expr_arena))
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UserDefinedNode {
    pub name: String,
    pub bytes: Vec<u8>,
}

pub trait FunctionRegistry {
    fn try_encode_scan(&self, _scan: &dyn AnonymousScan) -> PolarsResult<UserDefinedNode>;
    fn try_encode_udf(&self, _udf: &dyn DataFrameUdf, _buf: &mut Vec<u8>) -> PolarsResult<()>;
    fn try_decode_scan(
        &self,
        _node: &UserDefinedNode,
    ) -> PolarsResult<Option<Arc<dyn AnonymousScan>>>;
    fn try_decode_udf(
        &self,
        _name: &str,
        _bytes: &[u8],
    ) -> PolarsResult<Option<Arc<dyn DataFrameUdf>>>;
}
struct DefaultFunctionRegistry;

impl FunctionRegistry for DefaultFunctionRegistry {
    fn try_encode_scan(&self, _scan: &dyn AnonymousScan) -> PolarsResult<UserDefinedNode> {
        polars_bail!(InvalidOperation: "no default implementation for encoding scans")
    }

    fn try_encode_udf(&self, _udf: &dyn DataFrameUdf, _buf: &mut Vec<u8>) -> PolarsResult<()> {
        polars_bail!(InvalidOperation: "no default implementation for encoding udfs")
    }

    fn try_decode_scan(
        &self,
        _node: &UserDefinedNode,
    ) -> PolarsResult<Option<Arc<dyn AnonymousScan>>> {
        polars_bail!(InvalidOperation: "no default implementation for decoding scans")
    }

    fn try_decode_udf(
        &self,
        _name: &str,
        _bytes: &[u8],
    ) -> PolarsResult<Option<Arc<dyn DataFrameUdf>>> {
        polars_bail!(InvalidOperation: "no default implementation for decoding udfs")
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SerializableAggregate {
    name: String,
    apply: Option<Vec<u8>>,
    input: Box<SerializableLogicalPlan>,
    keys: Arc<Vec<Expr>>,
    aggs: Vec<Expr>,
    schema: SchemaRef,
    maintain_order: bool,
    options: Arc<GroupbyOptions>,
}
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    Aggregate(SerializableAggregate),
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
            SerializableLogicalPlan::Aggregate(SerializableAggregate {
                name,
                apply,
                input,
                keys,
                aggs,
                schema,
                maintain_order,
                options,
            }) => {
                let input = input.try_into_logical_plan(registry)?;
                let apply = apply
                    .map(|buf| registry.try_decode_udf(&name, &buf))
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

#[cfg(feature = "serde")]
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
