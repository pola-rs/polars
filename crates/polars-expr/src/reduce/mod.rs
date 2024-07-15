mod convert;
mod extrema;
mod mean;
mod sum;

use std::any::Any;

pub use convert::{can_convert_into_reduction, into_reduction};
use polars_core::prelude::*;

#[allow(dead_code)]
pub trait Reduction: Any + Send {
    fn init(&mut self);

    fn update(&mut self, batch: &Series) -> PolarsResult<()>;

    unsafe fn update_gathered(&mut self, batch: &Series, idx: &[IdxSize]) -> PolarsResult<()> {
        let batch = batch.take_unchecked_from_slice(idx);
        self.update(&batch)
    }

    fn combine(&mut self, other: &dyn Reduction) -> PolarsResult<()>;

    fn finalize(&mut self) -> PolarsResult<Scalar>;

    fn as_any(&self) -> &dyn Any;
}
