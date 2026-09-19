mod dispatch;
#[cfg(feature = "rolling_window_by")]
mod rolling_kernels;

pub use dispatch::*;
use polars_arrow::array::{ArrayRef, PrimitiveArray};
use polars_compute::rolling;
use polars_core::prelude::*;
use polars_defs::time::rolling::RollingOptionsDynamicWindow;
