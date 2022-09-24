use std::borrow::Cow;
use std::cell::Cell;
use std::fmt::Debug;
#[cfg(any(feature = "ipc", feature = "csv-file", feature = "parquet"))]
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use polars_core::prelude::*;

use crate::logical_plan::LogicalPlan::DataFrameScan;
use crate::prelude::*;
use crate::utils::{expr_to_leaf_column_names, get_single_leaf, has_expr, has_wildcard};

pub(crate) mod aexpr;
pub(crate) mod alp;
pub(crate) mod anonymous_scan;

mod apply;
mod builder;
pub(crate) mod conversion;
mod format;
mod functions;
pub(crate) mod iterator;
mod lit;
pub(crate) mod optimizer;
pub(crate) mod options;
mod projection;
mod schema;

pub use anonymous_scan::*;
pub use apply::*;
pub(crate) use builder::*;
pub use lit::*;
use polars_core::frame::explode::MeltArgs;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub(crate) use crate::logical_plan::functions::FunctionNode;

// Will be set/ unset in the fetch operation to communicate overwriting the number of rows to scan.
thread_local! {pub(crate) static FETCH_ROWS: Cell<Option<usize>> = Cell::new(None)}

#[derive(Clone, Copy, Debug)]
pub enum Context {
    /// Any operation that is done on groups
    Aggregation,
    /// Any operation that is done while projection/ selection of data
    Default,
}

// https://stackoverflow.com/questions/1031076/what-are-projection-and-selection
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LogicalPlan {
    #[cfg_attr(feature = "serde", serde(skip))]
    AnonymousScan {
        function: Arc<dyn AnonymousScan>,
        schema: SchemaRef,
        predicate: Option<Expr>,
        aggregate: Vec<Expr>,
        options: AnonymousScanOptions,
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
    /// Scan a CSV file
    #[cfg(feature = "csv-file")]
    CsvScan {
        path: PathBuf,
        schema: SchemaRef,
        options: CsvParserOptions,
        /// Filters at the scan level
        predicate: Option<Expr>,
        /// Aggregations at the scan level
        aggregate: Vec<Expr>,
    },
    #[cfg(feature = "parquet")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parquet")))]
    /// Scan a Parquet file
    ParquetScan {
        path: PathBuf,
        schema: SchemaRef,
        predicate: Option<Expr>,
        aggregate: Vec<Expr>,
        options: ParquetOptions,
    },
    #[cfg(feature = "ipc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ipc")))]
    IpcScan {
        path: PathBuf,
        schema: SchemaRef,
        options: IpcScanOptionsInner,
        predicate: Option<Expr>,
        aggregate: Vec<Expr>,
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
        options: GroupbyOptions,
    },
    /// Join operation
    Join {
        input_left: Box<LogicalPlan>,
        input_right: Box<LogicalPlan>,
        schema: SchemaRef,
        left_on: Vec<Expr>,
        right_on: Vec<Expr>,
        options: JoinOptions,
    },
    /// Adding columns to the table without a Join
    HStack {
        input: Box<LogicalPlan>,
        exprs: Vec<Expr>,
        schema: SchemaRef,
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
    /// An explode operation
    Explode {
        input: Box<LogicalPlan>,
        columns: Vec<String>,
        schema: SchemaRef,
    },
    /// Slice the table
    Slice {
        input: Box<LogicalPlan>,
        offset: i64,
        len: IdxSize,
    },
    /// A Melt operation
    Melt {
        input: Box<LogicalPlan>,
        args: Arc<MeltArgs>,
        schema: SchemaRef,
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
        err: Arc<Mutex<Option<PolarsError>>>,
    },
    /// This allows expressions to access other tables
    ExtContext {
        input: Box<LogicalPlan>,
        contexts: Vec<LogicalPlan>,
        schema: SchemaRef,
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
    #[cfg(test)]
    pub(crate) fn into_alp(self) -> (Node, Arena<ALogicalPlan>, Arena<AExpr>) {
        let mut lp_arena = Arena::with_capacity(16);
        let mut expr_arena = Arena::with_capacity(16);
        let root = to_alp(self, &mut expr_arena, &mut lp_arena).unwrap();
        (root, lp_arena, expr_arena)
    }
}

impl LogicalPlan {
    pub(crate) fn schema(&self) -> PolarsResult<Cow<'_, SchemaRef>> {
        use LogicalPlan::*;
        match self {
            #[cfg(feature = "python")]
            PythonScan { options } => Ok(Cow::Borrowed(&options.schema)),
            Union { inputs, .. } => inputs[0].schema(),
            Cache { input, .. } => input.schema(),
            Sort { input, .. } => input.schema(),
            Explode { schema, .. } => Ok(Cow::Borrowed(schema)),
            #[cfg(feature = "parquet")]
            ParquetScan { schema, .. } => Ok(Cow::Borrowed(schema)),
            #[cfg(feature = "ipc")]
            IpcScan { schema, .. } => Ok(Cow::Borrowed(schema)),
            DataFrameScan { schema, .. } => Ok(Cow::Borrowed(schema)),
            AnonymousScan { schema, .. } => Ok(Cow::Borrowed(schema)),
            Selection { input, .. } => input.schema(),
            #[cfg(feature = "csv-file")]
            CsvScan { schema, .. } => Ok(Cow::Borrowed(schema)),
            Projection { schema, .. } => Ok(Cow::Borrowed(schema)),
            LocalProjection { schema, .. } => Ok(Cow::Borrowed(schema)),
            Aggregate { schema, .. } => Ok(Cow::Borrowed(schema)),
            Join { schema, .. } => Ok(Cow::Borrowed(schema)),
            HStack { schema, .. } => Ok(Cow::Borrowed(schema)),
            Distinct { input, .. } => input.schema(),
            Slice { input, .. } => input.schema(),
            Melt { schema, .. } => Ok(Cow::Borrowed(schema)),
            MapFunction {
                input, function, ..
            } => {
                let input_schema = input.schema()?;
                match input_schema {
                    Cow::Owned(schema) => Ok(Cow::Owned(function.schema(&schema)?.into_owned())),
                    Cow::Borrowed(schema) => function.schema(schema),
                }
            }
            Error { err, .. } => {
                // We just take the error. The LogicalPlan should not be used anymore once this
                let mut err = err.lock();
                match err.take() {
                    Some(err) => Err(err),
                    None => Err(PolarsError::ComputeError(
                        "LogicalPlan already failed".into(),
                    )),
                }
            }
            ExtContext { schema, .. } => Ok(Cow::Borrowed(schema)),
        }
    }
    pub fn describe(&self) -> String {
        format!("{:#?}", self)
    }
}

#[cfg(test)]
mod test {
    use polars_core::df;
    use polars_core::prelude::*;

    use crate::prelude::*;
    use crate::tests::get_df;

    fn print_plans(lf: &LazyFrame) {
        println!("LOGICAL PLAN\n\n{}\n", lf.describe_plan());
        println!(
            "OPTIMIZED LOGICAL PLAN\n\n{}\n",
            lf.describe_optimized_plan().unwrap()
        );
    }

    #[test]
    fn test_lazy_arithmetic() {
        let df = get_df();
        let lf = df
            .lazy()
            .select(&[((col("sepal.width") * lit(100)).alias("super_wide"))])
            .sort("super_wide", SortOptions::default());

        print_plans(&lf);

        let new = lf.collect().unwrap();
        println!("{:?}", new);
        assert_eq!(new.height(), 7);
        assert_eq!(
            new.column("super_wide").unwrap().f64().unwrap().get(0),
            Some(300.0)
        );
    }

    #[test]
    fn test_lazy_logical_plan_filter_and_alias_combined() {
        let df = get_df();
        let lf = df
            .lazy()
            .filter(col("sepal.width").lt(lit(3.5)))
            .select(&[col("variety").alias("foo")]);

        print_plans(&lf);
        let df = lf.collect().unwrap();
        println!("{:?}", df);
    }

    #[test]
    fn test_lazy_logical_plan_schema() {
        let df = get_df();
        let lp = df
            .clone()
            .lazy()
            .select(&[col("variety").alias("foo")])
            .logical_plan;

        assert!(lp.schema().unwrap().get("foo").is_some());

        let lp = df
            .lazy()
            .groupby([col("variety")])
            .agg([col("sepal.width").min()])
            .logical_plan;
        assert!(lp.schema().unwrap().get("sepal.width").is_some());
    }

    #[test]
    fn test_lazy_logical_plan_join() {
        let left = df!("days" => &[0, 1, 2, 3, 4],
        "temp" => [22.1, 19.9, 7., 2., 3.],
        "rain" => &[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        .unwrap();

        let right = df!(
        "days" => &[1, 2],
        "rain" => &[0.1, 0.2]
        )
        .unwrap();

        // check if optimizations succeeds without selection
        {
            let lf = left
                .clone()
                .lazy()
                .left_join(right.clone().lazy(), col("days"), col("days"));

            print_plans(&lf);
            // implicitly checks logical plan == optimized logical plan
            let _df = lf.collect().unwrap();
        }

        // check if optimization succeeds with selection
        {
            let lf = left
                .clone()
                .lazy()
                .left_join(right.clone().lazy(), col("days"), col("days"))
                .select(&[col("temp")]);

            let _df = lf.collect().unwrap();
        }

        // check if optimization succeeds with selection of a renamed column due to the join
        {
            let lf = left
                .lazy()
                .left_join(right.lazy(), col("days"), col("days"))
                .select(&[col("temp"), col("rain_right")]);

            print_plans(&lf);
            let _df = lf.collect().unwrap();
        }
    }
}
