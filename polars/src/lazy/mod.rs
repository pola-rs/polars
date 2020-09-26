// All credits to Andy Grove and Ballista/ DataFusion / Apache Arrow

mod logical_plan;
mod physical_plan;

use crate::{
    lazy::{
        logical_plan::*,
        physical_plan::{expressions::*, planner::SimplePlanner, PhysicalExpr, PhysicalPlanner},
    },
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

    // physical plan see: datafusion/physical_plan/planner.rs.html#61-63

    #[test]
    fn plan_builder_simple() {
        let logical_plan =
            LogicalPlanBuilder::scan_csv("../data/iris.csv".into(), None, true, None)
                .filter(col("sepal.length").lt(lit(5)))
                .build();

        println!("{:?}", logical_plan);

        let planner = SimplePlanner {};
        let physical_plan = planner.create_physical_plan(&logical_plan).unwrap();
        println!("{:?}", physical_plan.execute());
    }
}
