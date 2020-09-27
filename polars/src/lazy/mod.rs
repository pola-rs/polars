// All credits to Andy Grove and Ballista/ DataFusion / Apache Arrow

pub mod frame;
mod logical_plan;
mod physical_plan;
pub(crate) mod prelude;

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
