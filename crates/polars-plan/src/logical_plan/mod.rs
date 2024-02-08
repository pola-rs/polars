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

use self::tree_format::{TreeFmtNode, TreeFmtVisitor};
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
pub(crate) struct ErrorStateUnsync {
    n_times: usize,
    err: PolarsError,
}

#[derive(Clone)]
pub struct ErrorState(pub(crate) Arc<Mutex<ErrorStateUnsync>>);

impl std::fmt::Debug for ErrorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let this = self.0.lock().unwrap();
        // Skip over the Arc<Mutex<ErrorStateUnsync>> and just print the fields we care
        // about. Technically this is misleading, but the insides of ErrorState are not
        // public, so this only affects authors of polars, not users (and the odds that
        // this affects authors is slim)
        f.debug_struct("ErrorState")
            .field("n_times", &this.n_times)
            .field("err", &this.err)
            .finish()
    }
}

impl From<PolarsError> for ErrorState {
    fn from(err: PolarsError) -> Self {
        Self(Arc::new(Mutex::new(ErrorStateUnsync { n_times: 0, err })))
    }
}

impl ErrorState {
    fn take(&self) -> PolarsError {
        let mut this = self.0.lock().unwrap();

        let ret_err = if this.n_times == 0 {
            this.err.wrap_msg(&|msg| msg.to_owned())
        } else {
            this.err.wrap_msg(&|msg| {
                let n_times = this.n_times;

                let plural_s;
                let was_were;

                if n_times == 1 {
                    plural_s = "";
                    was_were = "was"
                } else {
                    plural_s = "s";
                    was_were = "were";
                };
                format!(
                    "{msg}\n\nLogicalPlan had already failed with the above error; \
                     after failure, {n_times} additional operation{plural_s} \
                     {was_were} attempted on the LazyFrame",
                )
            })
        };
        this.n_times += 1;

        ret_err
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

    pub fn describe_tree_format(&self) -> String {
        let mut visitor = TreeFmtVisitor::default();
        TreeFmtNode::root_logical_plan(self).traverse(&mut visitor);
        format!("{visitor:#?}")
    }

    pub fn to_alp(self) -> PolarsResult<(Node, Arena<ALogicalPlan>, Arena<AExpr>)> {
        let mut lp_arena = Arena::with_capacity(16);
        let mut expr_arena = Arena::with_capacity(16);

        let node = to_alp(self, &mut expr_arena, &mut lp_arena)?;

        Ok((node, lp_arena, expr_arena))
    }
}
