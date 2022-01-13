#[cfg(feature = "parquet")]
mod io;
mod optimization_checks;
mod predicate_queries;
mod projection_queries;
mod queries;

use optimization_checks::*;

use polars_core::prelude::*;
use polars_io::prelude::*;
use std::io::Cursor;

use crate::functions::{argsort_by, pearson_corr};
use crate::logical_plan::iterator::ArenaLpIter;
use crate::logical_plan::optimizer::simplify_expr::SimplifyExprRule;
use crate::logical_plan::optimizer::stack_opt::{OptimizationRule, StackOptimizer};
use crate::prelude::*;
use polars_core::chunked_array::builder::get_list_builder;
use polars_core::df;
#[cfg(feature = "temporal")]
use polars_core::export::chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use std::iter::FromIterator;

static GLOB_PARQUET: &str = "../../examples/aggregate_multiple_files_in_chunks/datasets/*.parquet";
static GLOB_CSV: &str = "../../examples/aggregate_multiple_files_in_chunks/datasets/*.csv";
static GLOB_IPC: &str = "../../examples/aggregate_multiple_files_in_chunks/datasets/*.ipc";

fn scan_foods_csv() -> LazyFrame {
    let path = "../../examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv";
    LazyCsvReader::new(path.to_string()).finish().unwrap()
}

fn init_files() {
    for path in &[
        "../../examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv",
        "../../examples/aggregate_multiple_files_in_chunks/datasets/foods2.csv",
    ] {
        let out_path1 = path.replace(".csv", ".parquet");
        let out_path2 = path.replace(".csv", ".ipc");

        for out_path in [out_path1, out_path2] {
            if std::fs::metadata(&out_path).is_err() {
                let df = CsvReader::from_path(path).unwrap().finish().unwrap();

                if out_path.ends_with("parquet") {
                    let f = std::fs::File::create(&out_path).unwrap();
                    ParquetWriter::new(f)
                        .with_statistics(true)
                        .finish(&df)
                        .unwrap();
                } else {
                    let f = std::fs::File::create(&out_path).unwrap();
                    IpcWriter::new(f).finish(&df).unwrap();
                }
            }
        }
    }
}

#[cfg(feature = "parquet")]
fn scan_foods_parquet(parallel: bool) -> LazyFrame {
    init_files();
    let out_path =
        "../../examples/aggregate_multiple_files_in_chunks/datasets/foods1.parquet".into();

    let args = ScanArgsParquet {
        n_rows: None,
        cache: false,
        parallel,
        rechunk: true,
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
