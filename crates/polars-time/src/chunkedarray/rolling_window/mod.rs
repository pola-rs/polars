mod dispatch;
#[cfg(feature = "rolling_window_by")]
mod rolling_kernels;

use arrow::array::{ArrayRef, PrimitiveArray};
pub use dispatch::*;
use polars_compute::rolling;
use polars_core::prelude::*;
use polars_defs::time::rolling::RollingOptionsDynamicWindow;
