fn main(){
use polars::prelude::*;

let q = LazyCsvReader::new("docs/data/iris.csv")
    .has_header(true)
    .finish()?
    .filter(col("sepal_length").gt(lit(5)))
    .group_by(vec![col("species")])
    .agg([col("*").sum()]);

let df = q.collect();
}
