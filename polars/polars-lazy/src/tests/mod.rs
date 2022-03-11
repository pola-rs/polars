#[cfg(feature = "test")]
mod aggregations;
#[cfg(feature = "test")]
mod arity;
#[cfg(feature = "parquet")]
mod io;
#[cfg(feature = "test")]
mod logical;
#[cfg(feature = "test")]
mod optimization_checks;
#[cfg(feature = "test")]
mod predicate_queries;
#[cfg(feature = "test")]
mod projection_queries;
#[cfg(feature = "test")]
mod queries;

fn load_df() -> DataFrame {
    df!("a" => &[1, 2, 3, 4, 5],
                 "b" => &["a", "a", "b", "c", "c"],
                 "c" => &[1, 2, 3, 4, 5]
    )
    .unwrap()
}

use optimization_checks::*;
use std::sync::Mutex;

use polars_core::prelude::*;
use polars_io::prelude::*;
use std::io::Cursor;

use crate::dsl::{argsort_by, pearson_corr};
use crate::logical_plan::iterator::ArenaLpIter;
use crate::logical_plan::optimizer::simplify_expr::SimplifyExprRule;
use crate::logical_plan::optimizer::stack_opt::{OptimizationRule, StackOptimizer};
use crate::prelude::*;
use polars_core::chunked_array::builder::get_list_builder;
use polars_core::df;
#[cfg(feature = "temporal")]
use polars_core::export::chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use polars_core::export::lazy_static::lazy_static;
use std::iter::FromIterator;

static GLOB_PARQUET: &str = "../../examples/datasets/*.parquet";
static GLOB_CSV: &str = "../../examples/datasets/*.csv";
static GLOB_IPC: &str = "../../examples/datasets/*.ipc";
static FOODS_CSV: &str = "../../examples/datasets/foods1.csv";
static FOODS_IPC: &str = "../../examples/datasets/foods1.ipc";
static FOODS_PARQUET: &str = "../../examples/datasets/foods1.parquet";

lazy_static! {
    // needed prevent race conditions during test execution
    // for example with env vars set for tests
    // or global string cache resets.
    pub(crate) static ref SINGLE_LOCK: Mutex<()> = Mutex::new(());
}

fn scan_foods_csv() -> LazyFrame {
    LazyCsvReader::new(FOODS_CSV.to_string()).finish().unwrap()
}

fn scan_foods_ipc() -> LazyFrame {
    LazyFrame::scan_ipc(FOODS_IPC.to_string(), Default::default()).unwrap()
}

fn init_files() {
    for path in &[
        "../../examples/datasets/foods1.csv",
        "../../examples/datasets/foods2.csv",
    ] {
        let out_path1 = path.replace(".csv", ".parquet");
        let out_path2 = path.replace(".csv", ".ipc");

        for out_path in [out_path1, out_path2] {
            if std::fs::metadata(&out_path).is_err() {
                let mut df = CsvReader::from_path(path).unwrap().finish().unwrap();

                if out_path.ends_with("parquet") {
                    let f = std::fs::File::create(&out_path).unwrap();
                    ParquetWriter::new(f)
                        .with_statistics(true)
                        .finish(&mut df)
                        .unwrap();
                } else {
                    let f = std::fs::File::create(&out_path).unwrap();
                    IpcWriter::new(f).finish(&mut df).unwrap();
                }
            }
        }
    }
}

#[cfg(feature = "parquet")]
fn scan_foods_parquet(parallel: bool) -> LazyFrame {
    init_files();
    let out_path = FOODS_PARQUET.to_string();

    let args = ScanArgsParquet {
        n_rows: None,
        cache: false,
        parallel,
        rechunk: true,
        row_count: None,
    };
    LazyFrame::scan_parquet(out_path, args).unwrap()
}

pub(crate) fn fruits_cars() -> DataFrame {
    df!(
            "A"=> [1, 2, 3, 4, 5],
            "fruits"=> ["banana", "banana", "apple", "apple", "banana"],
            "B"=> [5, 4, 3, 2, 1],
            "cars"=> ["beetle", "audi", "beetle", "beetle", "beetle"]
    )
    .unwrap()
}

// physical plan see: datafusion/physical_plan/planner.rs.html#61-63

pub(crate) fn get_df() -> DataFrame {
    let s = r#"
"sepal.length","sepal.width","petal.length","petal.width","variety"
5.1,3.5,1.4,.2,"Setosa"
4.9,3,1.4,.2,"Setosa"
4.7,3.2,1.3,.2,"Setosa"
4.6,3.1,1.5,.2,"Setosa"
5,3.6,1.4,.2,"Setosa"
5.4,3.9,1.7,.4,"Setosa"
4.6,3.4,1.4,.3,"Setosa"
"#;

    let file = Cursor::new(s);

    let df = CsvReader::new(file)
        // we also check if infer schema ignores errors
        .infer_schema(Some(3))
        .has_header(true)
        .finish()
        .unwrap();
    df
}
