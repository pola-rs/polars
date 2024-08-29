mod convert;
mod len;
mod mean;
mod min_max;
#[cfg(feature = "propagate_nans")]
mod nan_min_max;
mod sum;

use std::any::Any;

pub use convert::into_reduction;
use polars_core::prelude::*;

pub trait Reduction: Send {
    /// Create a new reducer for this Reduction.
    fn new_reducer(&self) -> Box<dyn ReductionState>;
}

pub trait ReductionState: Any + Send {
    /// Adds the given series into the reduction.
    fn update(&mut self, batch: &Series) -> PolarsResult<()>;

    /// Adds the elements of the given series at the given indices into the reduction.
    ///
    /// # Safety
    /// Implementations may elide bound checks.
    unsafe fn update_gathered(&mut self, batch: &Series, idx: &[IdxSize]) -> PolarsResult<()> {
        let batch = batch.take_unchecked_from_slice(idx);
        self.update(&batch)
    }

    /// Combines this reduction with another.
    fn combine(&mut self, other: &dyn ReductionState) -> PolarsResult<()>;

    /// Returns a final result from the reduction.
    fn finalize(&self) -> PolarsResult<Scalar>;

    /// Returns this ReductionState as a dyn Any.
    fn as_any(&self) -> &dyn Any;
}
