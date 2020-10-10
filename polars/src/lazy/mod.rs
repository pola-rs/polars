//! Lazy API of Polars
//!
//! *Credits to the work of Andy Grove and Ballista/ DataFusion / Apache Arrow, which gave
//! this a huge kickstart.*
//!
//! The lazy api of Polars supports a subset of the eager api. Apart from the distributed compute,
//! it is very similar to [Apache Spark](https://spark.apache.org/). You write queries in a
//! domain specific language. These queries translate to a logical plan, which represent your query steps.
//! Before execution this logical plan is optimized and may change the order of operations if this will increase performance.
//! Or implicit type casts may be added such that execution of the query won't lead to a type error (if it can be resolved).
//!
//! The easiest way to get started is with the [LazyFrame](crate::lazy::frame::LazyFrame) struct.
//! The method's docstrings show some examples to get you up to speed.
//!
pub mod dsl;
pub mod frame;
mod logical_plan;
mod physical_plan;
pub(crate) mod prelude;
pub(crate) mod utils;

#[cfg(test)]
mod tests {
    use crate::lazy::prelude::*;
    use crate::prelude::*;
    use std::io::Cursor;

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

    #[test]
    fn plan_builder_simple() {
        let df = get_df();

        let logical_plan = LogicalPlanBuilder::from_existing_df(df)
            .filter(col("sepal.length").lt(lit(5)))
            .build();

        println!("{:?}", logical_plan);

        let planner = DefaultPlanner {};
        let physical_plan = planner.create_physical_plan(&logical_plan).unwrap();
        println!("{:?}", physical_plan.execute());
    }
}
