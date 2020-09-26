pub mod executors;
pub mod expressions;
pub mod planner;

use super::*;
use std::fmt::Debug;
use std::rc::Rc;

pub trait PhysicalPlanner {
    fn create_physical_plan(&self, logical_plan: &LogicalPlan) -> Result<Rc<dyn ExecutionPlan>>;
}

pub trait ExecutionPlan: Debug {
    fn schema(&self) -> SchemaRef {
        todo!()
    }
    fn execute(&self) -> Result<DataStructure>;
}

/// Implement this for Column, lt, eq, etc
pub trait PhysicalExpr: Debug {
    fn data_type(&self, input_schema: &Schema) -> Result<ArrowDataType>;
    fn evaluate(&self, ds: &DataStructure) -> Result<Series>;
}
