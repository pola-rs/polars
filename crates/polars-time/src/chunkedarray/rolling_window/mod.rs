use std::hash::{Hash, Hasher};

mod dispatch;
#[cfg(feature = "rolling_window_by")]
mod rolling_kernels;

use arrow::array::{ArrayRef, PrimitiveArray};
pub use dispatch::*;
use polars_compute::rolling;
use polars_compute::rolling::RollingFnParams;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "rolling_window_by", derive(PartialEq))]
pub struct RollingOptionsDynamicWindow {
    /// The length of the window.
    pub window_size: Duration,
    /// Amount of elements in the window that should be filled before computing a result.
    pub min_periods: usize,
    /// Which side windows should be closed.
    pub closed_window: ClosedWindow,
    /// Optional parameters for the rolling
    #[cfg_attr(any(feature = "serde", feature = "dsl-schema"), serde(default))]
    pub fn_params: Option<RollingFnParams>,
}

impl Hash for RollingOptionsDynamicWindow {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.window_size.hash(state);
        self.min_periods.hash(state);
        self.closed_window.hash(state);
    }
}
