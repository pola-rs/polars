mod convert;
mod extrema;
mod len;
mod mean;
mod sum;

use std::any::Any;

pub use convert::into_reduction;
use polars_core::prelude::*;

#[allow(dead_code)]
pub trait Reduction: Any + Send {
    // Creates a fresh reduction.
    fn init_dyn(&self) -> Box<dyn Reduction>;

    // Resets this reduction to the fresh initial state.
    fn reset(&mut self);

    fn update(&mut self, batch: &Series) -> PolarsResult<()>;

    /// # Safety
    /// Implementations may elide bound checks.
    unsafe fn update_gathered(&mut self, batch: &Series, idx: &[IdxSize]) -> PolarsResult<()> {
        let batch = batch.take_unchecked_from_slice(idx);
        self.update(&batch)
    }

    fn combine(&mut self, other: &dyn Reduction) -> PolarsResult<()>;

    fn finalize(&mut self) -> PolarsResult<Scalar>;

    fn as_any(&self) -> &dyn Any;
}
