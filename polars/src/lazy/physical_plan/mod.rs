pub mod executors;
pub mod expressions;
pub mod planner;

use super::*;
use std::fmt::Debug;
use std::rc::Rc;

pub enum ExprVal {
    Series(Series),
    Column(Vec<String>),
}

pub trait PhysicalPlanner {
    fn create_physical_plan(&self, logical_plan: &LogicalPlan) -> Result<Rc<dyn ExecutionPlan>>;
}

pub trait ExecutionPlan: Debug {
    fn schema(&self) -> SchemaRef {
        todo!()
    }
    fn execute(&self) -> Result<DataFrame>;
}

/// Implement this for Column, lt, eq, etc
pub trait PhysicalExpr: Debug {
    fn data_type(&self, _input_schema: &Schema) -> Result<ArrowDataType> {
        unimplemented!()
    }
    fn evaluate(&self, df: &DataFrame) -> Result<Series>;
}
