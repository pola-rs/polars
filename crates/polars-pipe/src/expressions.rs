use std::any::Any;

use polars_core::prelude::*;
use polars_plan::dsl::Expr;

use crate::operators::DataChunk;

pub trait PhysicalPipedExpr: Send + Sync {
    /// Take a `DataFrame` and produces a boolean `Series` that serves
    /// as a predicate mask
    fn evaluate(&self, chunk: &DataChunk, lazy_state: &dyn Any) -> PolarsResult<Series>;

    fn field(&self, input_schema: &Schema) -> PolarsResult<Field>;

    fn expression(&self) -> Expr;
}
