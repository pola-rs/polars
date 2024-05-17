mod convert;
mod extrema;
mod sum;

use std::any::Any;

use arrow::legacy::error::PolarsResult;
use polars_core::datatypes::Scalar;
use polars_core::prelude::Series;

#[allow(dead_code)]
trait Reduction: Any {
    fn init(&mut self);

    fn update(&mut self, batch: &Series) -> PolarsResult<()>;

    fn combine(&mut self, other: &dyn Reduction) -> PolarsResult<()>;

    fn finalize(&mut self) -> PolarsResult<Scalar>;

    fn as_any(&self) -> &dyn Any;
}
