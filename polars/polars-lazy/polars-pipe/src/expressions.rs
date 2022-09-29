use polars_core::prelude::*;

use crate::operators::DataChunk;

pub trait PhysicalPipedExpr: Send + Sync {
    /// Take a `DataFrame` and produces a boolean `Series` that serves
    /// as a predicate mask
    fn evaluate(&self, chunk: DataChunk) -> PolarsResult<Series>;
}
