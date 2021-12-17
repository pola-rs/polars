mod predicate_pushdown;
mod queries;
use polars_core::prelude::*;
use polars_io::prelude::*;
use std::io::Cursor;

use crate::functions::{argsort_by, pearson_corr};
use crate::logical_plan::iterator::ArenaLpIter;
use crate::logical_plan::optimizer::simplify_expr::SimplifyExprRule;
use crate::logical_plan::optimizer::stack_opt::{OptimizationRule, StackOptimizer};
use crate::prelude::*;
use polars_core::chunked_array::builder::get_list_builder;
#[cfg(feature = "temporal")]
use polars_core::export::chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use polars_core::{df, prelude::*};
use std::iter::FromIterator;

fn scan_foods_csv() -> LazyFrame {
    let path = "../../examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv";
    LazyCsvReader::new(path.to_string()).finish().unwrap()
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
