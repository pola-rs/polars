mod dispatch;
#[cfg(feature = "rolling_window_by")]
mod rolling_kernels;

use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::legacy::kernels::rolling;
pub use dispatch::*;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollingOptionsDynamicWindow {
    /// The length of the window.
    pub window_size: Duration,
    /// Amount of elements in the window that should be filled before computing a result.
    pub min_periods: usize,
    /// Which side windows should be closed.
    pub closed_window: ClosedWindow,
    /// Optional parameters for the rolling function
    #[cfg_attr(feature = "serde", serde(skip))]
    pub fn_params: DynArgs,
}

#[cfg(feature = "rolling_window_by")]
impl PartialEq for RollingOptionsDynamicWindow {
    fn eq(&self, other: &Self) -> bool {
        self.window_size == other.window_size
            && self.min_periods == other.min_periods
            && self.closed_window == other.closed_window
            && self.fn_params.is_none()
            && other.fn_params.is_none()
    }
}
