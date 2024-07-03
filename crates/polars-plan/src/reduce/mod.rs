mod convert;
mod extrema;
mod sum;
mod nth;

use std::any::Any;

use arrow::legacy::error::PolarsResult;
use polars_core::datatypes::Scalar;
use polars_core::prelude::Series;
use polars_utils::IdxSize;

#[allow(dead_code)]
trait Reduction: Any {
    fn init(&mut self);

    fn update(&mut self, batch: &Series) -> PolarsResult<()>;

    unsafe fn update_gathered(&mut self, batch: &Series, idx: &[IdxSize]) -> PolarsResult<()> {
        let batch= batch.take_unchecked_from_slice(idx);
        self.update(&batch)
    }

    fn combine(&mut self, other: &dyn Reduction) -> PolarsResult<()>;

    fn finalize(&mut self) -> PolarsResult<Scalar>;

    fn as_any(&self) -> &dyn Any;
}
