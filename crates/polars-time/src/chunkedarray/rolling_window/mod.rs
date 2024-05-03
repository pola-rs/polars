mod dispatch;
mod rolling_kernels;

use arrow::array::{Array, ArrayRef, PrimitiveArray};
use arrow::legacy::kernels::rolling;
pub use dispatch::*;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    #[cfg_attr(feature = "serde", serde(skip))]
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

#[cfg(feature = "rolling_window")]
impl PartialEq for RollingOptions {
    fn eq(&self, other: &Self) -> bool {
        self.window_size == other.window_size
            && self.min_periods == other.min_periods
            && self.weights == other.weights
            && self.center == other.center
            && self.by == other.by
            && self.closed_window == other.closed_window
            && self.fn_params.is_none()
            && other.fn_params.is_none()
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
        RollingOptionsImpl {
            window_size: options.window_size,
            min_periods: options.min_periods,
            weights: options.weights,
            center: options.center,
            by: None,
            tu: None,
            tz: None,
            closed_window: options.closed_window,
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

impl<'a> TryFrom<RollingOptionsImpl<'a>> for RollingOptionsFixedWindow {
    type Error = PolarsError;
    fn try_from(options: RollingOptionsImpl<'a>) -> PolarsResult<Self> {
        polars_ensure!(
            options.window_size.parsed_int,
            InvalidOperation: "if `window_size` is a temporal window (e.g. '1d', '2h, ...), then the `by` argument must be passed"
        );
        polars_ensure!(
            options.closed_window.is_none(),
            InvalidOperation: "`closed_window` is not supported for fixed window size rolling aggregations, \
            consider using DataFrame.rolling for greater flexibility",
        );
        let window_size = options.window_size.nanoseconds() as usize;
        check_input(window_size, options.min_periods)?;
        Ok(RollingOptionsFixedWindow {
            window_size,
            min_periods: options.min_periods,
            weights: options.weights,
            center: options.center,
            fn_params: options.fn_params,
        })
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

#[derive(Clone)]
pub struct RollingOptionsDynamicWindow<'a> {
    /// The length of the window.
    pub window_size: Duration,
    /// Amount of elements in the window that should be filled before computing a result.
    pub min_periods: usize,
    pub by: &'a [i64],
    pub tu: Option<TimeUnit>,
    pub tz: Option<&'a TimeZone>,
    pub closed_window: ClosedWindow,
    pub fn_params: DynArgs,
}

impl<'a> TryFrom<RollingOptionsImpl<'a>> for RollingOptionsDynamicWindow<'a> {
    type Error = PolarsError;
    fn try_from(options: RollingOptionsImpl<'a>) -> PolarsResult<Self> {
        let duration = options.window_size;
        polars_ensure!(duration.duration_ns() > 0 && !duration.negative, ComputeError:"window size should be strictly positive");
        polars_ensure!(
            options.weights.is_none(),
            InvalidOperation: "`weights` is not supported in 'rolling_*(..., by=...)' expression"
        );
        polars_ensure!(
           !options.window_size.parsed_int,
           InvalidOperation: "if `by` argument is passed, then `window_size` must be a temporal window (e.g. '1d' or '2h', not '3i')"
        );
        Ok(RollingOptionsDynamicWindow {
            window_size: options.window_size,
            min_periods: options.min_periods,
            by: options.by.expect("by must have been set to get here"),
            tu: options.tu,
            tz: options.tz,
            closed_window: options.closed_window.unwrap_or(ClosedWindow::Right),
            fn_params: options.fn_params,
        })
    }
}
