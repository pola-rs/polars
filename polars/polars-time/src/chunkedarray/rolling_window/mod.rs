mod floats;
mod ints;
#[cfg(feature = "rolling_window")]
mod rolling_kernels;

#[cfg(feature = "rolling_window")]
use std::convert::TryFrom;
use std::ops::SubAssign;

#[cfg(feature = "rolling_window")]
use arrow::array::{Array, PrimitiveArray};
use polars_arrow::data_types::IsFloat;
#[cfg(feature = "rolling_window")]
use polars_arrow::export::arrow;
#[cfg(feature = "rolling_window")]
use polars_arrow::kernels::rolling;
#[cfg(feature = "rolling_window")]
use polars_arrow::prelude::QuantileInterpolOptions;
use polars_core::prelude::*;

#[cfg(feature = "rolling_window")]
use crate::prelude::*;
use crate::series::WrapFloat;

#[derive(Clone)]
#[cfg(feature = "rolling_window")]
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
}

#[cfg(feature = "rolling_window")]
impl Default for RollingOptions {
    fn default() -> Self {
        RollingOptions {
            window_size: Duration::parse("3i"),
            min_periods: 1,
            weights: None,
            center: false,
            by: None,
            closed_window: None,
        }
    }
}

#[derive(Clone)]
#[cfg(feature = "rolling_window")]
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
    pub closed_window: Option<ClosedWindow>,
}

#[cfg(feature = "rolling_window")]
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
            closed_window: None,
        }
    }
}

#[cfg(feature = "rolling_window")]
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
        }
    }
}

#[cfg(feature = "rolling_window")]
impl Default for RollingOptionsImpl<'static> {
    fn default() -> Self {
        RollingOptionsImpl {
            window_size: Duration::parse("3i"),
            min_periods: 1,
            weights: None,
            center: false,
            by: None,
            tu: None,
            closed_window: None,
        }
    }
}

#[cfg(feature = "rolling_window")]
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
        }
    }
}

#[cfg(not(feature = "rolling_window"))]
pub trait RollingAgg {}

#[cfg(feature = "rolling_window")]
pub trait RollingAgg {
    /// Apply a rolling mean (moving mean) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their mean.
    fn rolling_mean(&self, options: RollingOptionsImpl) -> PolarsResult<Series>;

    /// Apply a rolling sum (moving sum) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their sum.
    fn rolling_sum(&self, options: RollingOptionsImpl) -> PolarsResult<Series>;

    /// Apply a rolling min (moving min) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their min.
    fn rolling_min(&self, options: RollingOptionsImpl) -> PolarsResult<Series>;

    /// Apply a rolling max (moving max) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their max.
    fn rolling_max(&self, options: RollingOptionsImpl) -> PolarsResult<Series>;

    /// Apply a rolling median (moving median) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be weighted according to the `weights` vector.
    fn rolling_median(&self, options: RollingOptionsImpl) -> PolarsResult<Series>;

    /// Apply a rolling quantile (moving quantile) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be weighted according to the `weights` vector.
    fn rolling_quantile(
        &self,
        quantile: f64,
        interpolation: QuantileInterpolOptions,
        options: RollingOptionsImpl,
    ) -> PolarsResult<Series>;

    /// Apply a rolling var (moving var) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their var.
    #[cfg(feature = "rolling_window")]
    fn rolling_var(&self, options: RollingOptionsImpl) -> PolarsResult<Series>;

    /// Apply a rolling std (moving std) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their std.
    fn rolling_std(&self, options: RollingOptionsImpl) -> PolarsResult<Series>;
}

/// utility
#[cfg(feature = "rolling_window")]
fn check_input(window_size: usize, min_periods: usize) -> PolarsResult<()> {
    if min_periods > window_size {
        Err(PolarsError::ComputeError(
            "`windows_size` should be >= `min_periods`".into(),
        ))
    } else {
        Ok(())
    }
}

#[cfg(feature = "rolling_window")]
#[allow(clippy::type_complexity)]
fn rolling_agg<T>(
    ca: &ChunkedArray<T>,
    options: RollingOptionsImpl,
    rolling_agg_fn: &dyn Fn(&[T::Native], usize, usize, bool, Option<&[f64]>) -> ArrayRef,
    rolling_agg_fn_nulls: &dyn Fn(
        &PrimitiveArray<T::Native>,
        usize,
        usize,
        bool,
        Option<&[f64]>,
    ) -> ArrayRef,
    rolling_agg_fn_dynamic: Option<
        &dyn Fn(&[T::Native], Duration, Duration, &[i64], ClosedWindow, TimeUnit) -> ArrayRef,
    >,
) -> PolarsResult<Series>
where
    T: PolarsNumericType,
{
    if ca.is_empty() {
        return Ok(Series::new_empty(ca.name(), ca.dtype()));
    }
    let ca = ca.rechunk();

    let arr = ca.downcast_iter().next().unwrap();
    // "5i" is a window size of 5, e.g. fixed
    let arr = if options.window_size.parsed_int {
        let options: RollingOptionsFixedWindow = options.into();
        check_input(options.window_size, options.min_periods)?;
        let ca = ca.rechunk();

        match ca.null_count() {
            0 => rolling_agg_fn(
                arr.values().as_slice(),
                options.window_size,
                options.min_periods,
                options.center,
                options.weights.as_deref(),
            ),
            _ => rolling_agg_fn_nulls(
                arr,
                options.window_size,
                options.min_periods,
                options.center,
                options.weights.as_deref(),
            ),
        }
    } else {
        if arr.null_count() > 0 {
            panic!("'rolling by' not yet supported for series with null values, consider using 'groupby_rolling'")
        }
        let values = arr.values().as_slice();
        let duration = options.window_size;
        let tu = options.tu.unwrap();
        let by = options.by.unwrap();
        let closed_window = options.closed_window.expect("closed window  must be set");
        let mut offset = duration;
        offset.negative = true;
        let func = rolling_agg_fn_dynamic.expect(
            "'rolling by' not yet supported for this expression, consider using 'groupby_rolling'",
        );

        func(values, duration, offset, by, closed_window, tu)
    };
    Series::try_from((ca.name(), arr))
}
