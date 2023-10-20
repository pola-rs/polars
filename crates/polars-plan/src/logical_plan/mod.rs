use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use polars_core::prelude::*;
#[cfg(any(feature = "cloud", feature = "parquet"))]
use polars_io::cloud::CloudOptions;

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
pub(super) mod hive;
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

#[cfg(any(
    feature = "ipc",
    feature = "parquet",
    feature = "csv",
    feature = "cse",
    feature = "json"
))]
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
pub(crate) enum ErrorStateEncounters {
    NotYetEncountered {
        err: PolarsError,
    },
    AlreadyEncountered {
        n_times: usize,
        orig_err: PolarsError,
    },
}

#[derive(Clone)]
pub struct ErrorState(pub(crate) Arc<Mutex<ErrorStateEncounters>>);

impl std::fmt::Debug for ErrorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Skip over the Arc<Mutex<_>> and just print the ErrorStateEncounters
        f.debug_tuple("ErrorState")
            .field(&self.0.lock().unwrap())
            .finish()
    }
}

impl From<PolarsError> for ErrorState {
    fn from(err: PolarsError) -> Self {
        Self(Arc::new(Mutex::new(
            ErrorStateEncounters::NotYetEncountered { err },
        )))
    }
}

impl ErrorState {
    fn take(&self) -> PolarsError {
        let mut curr_err = self.0.lock().unwrap();

        match &mut *curr_err {
            ErrorStateEncounters::NotYetEncountered { err: polars_err } => {
                let orig_err = polars_err.wrap_msg(&str::to_owned);

                let prev_err = std::mem::replace(
                    &mut *curr_err,
                    ErrorStateEncounters::AlreadyEncountered {
                        n_times: 1,
                        orig_err,
                    },
                );
                // Since we're in this branch, we know err was a NotYetEncountered
                match prev_err {
                    ErrorStateEncounters::NotYetEncountered { err } => err,
                    ErrorStateEncounters::AlreadyEncountered { .. } => unreachable!(),
                }
            },
            ErrorStateEncounters::AlreadyEncountered { n_times, orig_err } => {
                let err = orig_err.wrap_msg(&|msg| {
                    format!(
                        "LogicalPlan already failed (depth: {}) with error: '{}'",
                        n_times, msg
                    )
                });

                *n_times += 1;
                err
            },
        }
    }
}

// https://stackoverflow.com/questions/1031076/what-are-projection-and-selection
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LogicalPlan {
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
        paths: Arc<[PathBuf]>,
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
    /// Horizontal concatenation of multiple plans
    HConcat {
        inputs: Vec<LogicalPlan>,
        schema: SchemaRef,
        options: HConcatOptions,
    },
    /// Catches errors and throws them later
    #[cfg_attr(feature = "serde", serde(skip))]
    Error {
        input: Box<LogicalPlan>,
        err: ErrorState,
    },
    /// This allows expressions to access other tables
    ExtContext {
        input: Box<LogicalPlan>,
        contexts: Vec<LogicalPlan>,
        schema: SchemaRef,
    },
    Sink {
        input: Box<LogicalPlan>,
        payload: SinkType,
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
