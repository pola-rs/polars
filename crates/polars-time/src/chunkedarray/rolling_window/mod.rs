mod dispatch;
mod rolling_kernels;

use std::convert::TryFrom;

use arrow::array::{Array, PrimitiveArray};
use arrow::legacy::kernels::rolling;
pub use dispatch::*;
use polars_core::prelude::*;

use crate::prelude::*;

#[derive(Clone)]
pub struct RollingOptions {
    /// The length of the window.
    pub window_size: Duration,
    /// Amount of elements in the window that should be filled before computing a result.
    pub min_periods: usize,
    /// An optional slice with the same length as the window that will be multiplied
    ///              elementwise with the values in the window.
    pub weights: Option<Vec<f64>>,
    /// Set the labels at the center of the window.
    pub center: bool,
    /// Compute the rolling aggregates with a window defined by a time column
    pub by: Option<String>,
    /// The closed window of that time window if given
    pub closed_window: Option<ClosedWindow>,
    /// Optional parameters for the rolling function
    pub fn_params: DynArgs,
    /// Warn if data is not known to be sorted by `by` column (if passed)
    pub warn_if_unsorted: bool,
}

impl Default for RollingOptions {
    fn default() -> Self {
        RollingOptions {
            window_size: Duration::parse("3i"),
            min_periods: 1,
            weights: None,
            center: false,
            by: None,
            closed_window: None,
            fn_params: None,
            warn_if_unsorted: true,
        }
    }
}

#[derive(Clone)]
pub struct RollingOptionsImpl<'a> {
    /// The length of the window.
    pub window_size: Duration,
    /// Amount of elements in the window that should be filled before computing a result.
    pub min_periods: usize,
    /// An optional slice with the same length as the window that will be multiplied
    ///              elementwise with the values in the window.
    pub weights: Option<Vec<f64>>,
    /// Set the labels at the center of the window.
    pub center: bool,
    pub by: Option<&'a [i64]>,
    pub tu: Option<TimeUnit>,
    pub tz: Option<&'a TimeZone>,
    pub closed_window: Option<ClosedWindow>,
    pub fn_params: DynArgs,
}

impl From<RollingOptions> for RollingOptionsImpl<'static> {
    fn from(options: RollingOptions) -> Self {
        let window_size = options.window_size;
        assert!(
            window_size.parsed_int,
            "should be fixed integer window size at this point"
        );

        RollingOptionsImpl {
            window_size,
            min_periods: options.min_periods,
            weights: options.weights,
            center: options.center,
            by: None,
            tu: None,
            tz: None,
            closed_window: None,
            fn_params: options.fn_params,
        }
    }
}

impl From<RollingOptions> for RollingOptionsFixedWindow {
    fn from(options: RollingOptions) -> Self {
        let window_size = options.window_size;
        assert!(
            window_size.parsed_int,
            "should be fixed integer window size at this point"
        );

        RollingOptionsFixedWindow {
            window_size: window_size.nanoseconds() as usize,
            min_periods: options.min_periods,
            weights: options.weights,
            center: options.center,
            fn_params: options.fn_params,
        }
    }
}

impl Default for RollingOptionsImpl<'static> {
    fn default() -> Self {
        RollingOptionsImpl {
            window_size: Duration::parse("3i"),
            min_periods: 1,
            weights: None,
            center: false,
            by: None,
            tu: None,
            tz: None,
            closed_window: None,
            fn_params: None,
        }
    }
}

impl<'a> From<RollingOptionsImpl<'a>> for RollingOptionsFixedWindow {
    fn from(options: RollingOptionsImpl<'a>) -> Self {
        let window_size = options.window_size;
        assert!(
            window_size.parsed_int,
            "should be fixed integer window size at this point"
        );

        RollingOptionsFixedWindow {
            window_size: window_size.nanoseconds() as usize,
            min_periods: options.min_periods,
            weights: options.weights,
            center: options.center,
            fn_params: options.fn_params,
        }
    }
}

/// utility
fn check_input(window_size: usize, min_periods: usize) -> PolarsResult<()> {
    polars_ensure!(
        min_periods <= window_size,
        ComputeError: "`min_periods` should be <= `window_size`",
    );
    Ok(())
}
