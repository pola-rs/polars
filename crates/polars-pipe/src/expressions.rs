use polars_core::prelude::*;
use polars_expr::state::ExecutionState;
use polars_io::predicates::PhysicalIoExpr;
use polars_plan::dsl::Expr;

use crate::operators::DataChunk;

pub trait PhysicalPipedExpr: PhysicalIoExpr + Send + Sync {
    /// Take a [`DataFrame`] and produces a boolean [`Series`] that serves
    /// as a predicate mask
    fn evaluate(&self, chunk: &DataChunk, lazy_state: &ExecutionState) -> PolarsResult<Series>;

    fn field(&self, input_schema: &Schema) -> PolarsResult<Field>;

    fn expression(&self) -> Expr;
}
