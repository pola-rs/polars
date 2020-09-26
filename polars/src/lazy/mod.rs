// All credits to Andy Grove and Ballista/ DataFusion / Apache Arrow

mod logical_plan;
mod physical_plan;

pub(crate) use crate::{
    lazy::{logical_plan::*, physical_plan::expressions::*},
    prelude::*,
};
use arrow::datatypes::SchemaRef;

#[derive(Debug)]
pub enum DataStructure {
    Series(Series),
    DataFrame(DataFrame),
}

impl From<Series> for DataStructure {
    fn from(s: Series) -> Self {
        DataStructure::Series(s)
    }
}

impl From<DataFrame> for DataStructure {
    fn from(df: DataFrame) -> Self {
        DataStructure::DataFrame(df)
    }
}

impl DataStructure {
    pub fn series_ref(&self) -> Result<&Series> {
        if let DataStructure::Series(series) = self {
            Ok(series)
        } else {
            Err(PolarsError::DataTypeMisMatch)
        }
    }

    pub fn df_ref(&self) -> Result<&DataFrame> {
        if let DataStructure::DataFrame(df) = self {
            Ok(&df)
        } else {
            Err(PolarsError::DataTypeMisMatch)
        }
    }

    pub fn len(&self) -> usize {
        match self {
            DataStructure::Series(s) => s.len(),
            DataStructure::DataFrame(df) => df.height(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::physical_plan::{planner::SimplePlanner, PhysicalPlanner};
    use std::io::Cursor;

    // physical plan see: datafusion/physical_plan/planner.rs.html#61-63

    #[test]
    fn plan_builder_simple() {
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

        let logical_plan = LogicalPlanBuilder::dataframe(df)
            .filter(col("sepal.length").lt(lit(5)))
            .build();

        println!("{:?}", logical_plan);

        let planner = SimplePlanner {};
        let physical_plan = planner.create_physical_plan(&logical_plan).unwrap();
        println!("{:?}", physical_plan.execute());
    }
}
