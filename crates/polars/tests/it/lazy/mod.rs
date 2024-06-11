mod aggregation;
#[cfg(feature = "cse")]
mod cse;
mod cwc;
mod explodes;
mod expressions;
mod exprs;
mod folds;
mod functions;
mod group_by;
mod group_by_dynamic;
mod predicate_queries;
mod projection_queries;
mod queries;
mod schema;

use polars::prelude::*;

pub(crate) fn fruits_cars() -> DataFrame {
    df!(
            "A"=> [1, 2, 3, 4, 5],
            "fruits"=> ["banana", "banana", "apple", "apple", "banana"],
            "B"=> [5, 4, 3, 2, 1],
            "cars"=> ["beetle", "audi", "beetle", "beetle", "beetle"]
    )
    .unwrap()
}
