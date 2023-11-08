mod aggregations;
mod arity;
#[cfg(all(feature = "strings", feature = "cse"))]
mod cse;
#[cfg(feature = "parquet")]
mod io;
mod logical;
mod optimization_checks;
mod predicate_queries;
mod projection_queries;
mod queries;
#[cfg(feature = "streaming")]
mod streaming;
#[cfg(all(feature = "strings", feature = "cse"))]
mod tpch;

fn get_arenas() -> (Arena<AExpr>, Arena<ALogicalPlan>) {
    let expr_arena = Arena::with_capacity(16);
    let lp_arena = Arena::with_capacity(8);
    (expr_arena, lp_arena)
}

fn load_df() -> DataFrame {
    df!("a" => &[1, 2, 3, 4, 5],
                 "b" => &["a", "a", "b", "c", "c"],
                 "c" => &[1, 2, 3, 4, 5]
    )
    .unwrap()
}

use std::io::Cursor;
use std::iter::FromIterator;

use optimization_checks::*;
use polars_core::chunked_array::builder::get_list_builder;
use polars_core::df;
#[cfg(feature = "temporal")]
use polars_core::export::chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use polars_core::prelude::*;
#[cfg(feature = "parquet")]
pub(crate) use polars_core::SINGLE_LOCK;
use polars_io::prelude::*;
use polars_plan::logical_plan::{
    ArenaLpIter, OptimizationRule, SimplifyExprRule, StackOptimizer, TypeCoercionRule,
};

use crate::dsl::pearson_corr;
use crate::prelude::*;

#[cfg(feature = "parquet")]
static GLOB_PARQUET: &str = "../../examples/datasets/*.parquet";
#[cfg(feature = "csv")]
static GLOB_CSV: &str = "../../examples/datasets/*.csv";
#[cfg(feature = "ipc")]
static GLOB_IPC: &str = "../../examples/datasets/*.ipc";
#[cfg(feature = "parquet")]
static FOODS_PARQUET: &str = "../../examples/datasets/foods1.parquet";
#[cfg(feature = "csv")]
static FOODS_CSV: &str = "../../examples/datasets/foods1.csv";
#[cfg(feature = "ipc")]
static FOODS_IPC: &str = "../../examples/datasets/foods1.ipc";

#[cfg(feature = "csv")]
fn scan_foods_csv() -> LazyFrame {
    LazyCsvReader::new(FOODS_CSV).finish().unwrap()
}

#[cfg(feature = "ipc")]
fn scan_foods_ipc() -> LazyFrame {
    init_files();
    LazyFrame::scan_ipc(FOODS_IPC, Default::default()).unwrap()
}

#[cfg(any(feature = "ipc", feature = "parquet"))]
fn init_files() {
    for path in &[
        "../../examples/datasets/foods1.csv",
        "../../examples/datasets/foods2.csv",
    ] {
        for ext in [".parquet", ".ipc", ".ndjson"] {
            let out_path = path.replace(".csv", ext);

            if std::fs::metadata(&out_path).is_err() {
                let mut df = CsvReader::from_path(path).unwrap().finish().unwrap();
                let f = std::fs::File::create(&out_path).unwrap();

                match ext {
                    ".parquet" => {
                        #[cfg(feature = "parquet")]
                        {
                            ParquetWriter::new(f)
                                .with_statistics(true)
                                .finish(&mut df)
                                .unwrap();
                        }
                    },
                    ".ipc" => {
                        IpcWriter::new(f).finish(&mut df).unwrap();
                    },
                    ".ndjson" => {
                        #[cfg(feature = "json")]
                        {
                            JsonWriter::new(f).finish(&mut df).unwrap()
                        }
                    },
                    _ => panic!(),
                }
            }
        }
    }
}

#[cfg(feature = "parquet")]
fn scan_foods_parquet(parallel: bool) -> LazyFrame {
    init_files();
    let out_path = FOODS_PARQUET;
    let parallel = if parallel {
        ParallelStrategy::Auto
    } else {
        ParallelStrategy::None
    };

    let args = ScanArgsParquet {
        n_rows: None,
        cache: false,
        parallel,
        rechunk: true,
        ..Default::default()
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
