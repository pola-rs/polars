pub mod executors;
pub mod expressions;
pub mod planner;

use crate::prelude::*;
use ahash::RandomState;
use polars_core::prelude::*;
use polars_io::PhysicalIoExpr;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub enum ExprVal {
    Series(Series),
    Column(Vec<String>),
}

pub trait PhysicalPlanner {
    fn create_physical_plan(&self, logical_plan: LogicalPlan) -> Result<Box<dyn Executor>>;
}

// Executor are the executors of the physical plan and produce DataFrames. They
// combine physical expressions, which produce Series.

/// Executors will evaluate physical expressions and collect them in a DataFrame.
pub trait Executor: Send + Sync {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame>;
}

pub(crate) type Cache = Arc<Mutex<HashMap<String, DataFrame, RandomState>>>;

/// Take a DataFrame and evaluate the expressions.
/// Implement this for Column, lt, eq, etc
pub trait PhysicalExpr: Send + Sync {
    fn as_expression(&self) -> &Expr {
        // for instance not needed for aggregations (for now)
        unimplemented!()
    }

    /// Take a DataFrame and evaluate the expression.
    fn evaluate(&self, df: &DataFrame) -> Result<Series>;

    /// Get the output field of this expr
    fn to_field(&self, input_schema: &Schema) -> Result<Field>;

    fn as_agg_expr(&self) -> Result<&dyn AggPhysicalExpr> {
        let e = self.as_expression();
        Err(PolarsError::InvalidOperation(
            format!("{:?} is not an agg expression", e).into(),
        ))
    }
}

trait ToPhysicalIoExpr {
    fn into_physical_io_expr(self) -> Arc<dyn PhysicalIoExpr>;
}

pub struct PhysicalIoHelper {
    expr: Arc<dyn PhysicalExpr>,
}

impl PhysicalIoHelper {
    fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        PhysicalIoHelper { expr }
    }
}

impl PhysicalIoExpr for PhysicalIoHelper {
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        self.expr.evaluate(df)
    }
}

impl PhysicalIoExpr for dyn PhysicalExpr {
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        <Self as PhysicalExpr>::evaluate(self, df)
    }
}

pub trait AggPhysicalExpr {
    fn evaluate(&self, df: &DataFrame, groups: &[(u32, Vec<u32>)]) -> Result<Option<Series>>;

    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &[(u32, Vec<u32>)],
    ) -> Result<Option<Vec<Series>>> {
        // we return a vec, such that an implementor can return more information, such as a sum and count.
        self.evaluate(df, groups).map(|opt| opt.map(|s| vec![s]))
    }

    fn evaluate_partitioned_final(
        &self,
        final_df: &DataFrame,
        groups: &[(u32, Vec<u32>)],
    ) -> Result<Option<Series>> {
        self.evaluate(final_df, groups)
    }
}
