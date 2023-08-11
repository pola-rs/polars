use num::pow::Pow;
use num::Float;
use polars_core::export::num;

use super::*;

#[cfg(not(feature = "rolling_window"))]
impl<T> RollingAgg for WrapFloat<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Pow<T::Native, Output = T::Native> + Float,
    ChunkedArray<T>: IntoSeries,
{
}

#[cfg(feature = "rolling_window")]
impl<T> RollingAgg for WrapFloat<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Pow<T::Native, Output = T::Native> + Float,
    ChunkedArray<T>: IntoSeries,
{
    /// Apply a rolling mean (moving mean) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their mean.
    fn rolling_mean(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        rolling_agg(
            &self.0,
            options,
            &rolling::no_nulls::rolling_mean,
            &rolling::nulls::rolling_mean,
            Some(&super::rolling_kernels::no_nulls::rolling_mean),
        )
    }

    /// Apply a rolling sum (moving sum) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their sum.
    fn rolling_sum(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        rolling_agg(
            &self.0,
            options,
            &rolling::no_nulls::rolling_sum,
            &rolling::nulls::rolling_sum,
            Some(&super::rolling_kernels::no_nulls::rolling_sum),
        )
    }

    /// Apply a rolling min (moving min) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their min.
    fn rolling_min(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        rolling_agg(
            &self.0,
            options,
            &rolling::no_nulls::rolling_min,
            &rolling::nulls::rolling_min,
            Some(&super::rolling_kernels::no_nulls::rolling_min),
        )
    }

    /// Apply a rolling max (moving max) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their max.
    fn rolling_max(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        rolling_agg(
            &self.0,
            options,
            &rolling::no_nulls::rolling_max,
            &rolling::nulls::rolling_max,
            Some(&super::rolling_kernels::no_nulls::rolling_max),
        )
    }

    /// Apply a rolling median (moving median) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be weighted according to the `weights` vector.
    fn rolling_median(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        // At the last possible second, right before we do computations, make sure we're using the
        // right quantile parameters to get a median. This also lets us have the convenience of
        // calling `rolling_median` from Rust without a bunch of dedicated functions that just call
        // out to the `rolling_quantile` anyway.
        let mut options = options.clone();
        options.fn_params = Some(Arc::new(RollingQuantileParams {
            prob: 0.5,
            interpol: QuantileInterpolOptions::Linear,
        }) as Arc<dyn std::any::Any + Send + Sync>);
        rolling_agg(
            &self.0,
            options,
            &rolling::no_nulls::rolling_quantile,
            &rolling::nulls::rolling_quantile,
            Some(&super::rolling_kernels::no_nulls::rolling_quantile),
        )
    }

    /// Apply a rolling quantile (moving quantile) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be weighted according to the `weights` vector.
    fn rolling_quantile(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        rolling_agg(
            &self.0,
            options,
            &rolling::no_nulls::rolling_quantile,
            &rolling::nulls::rolling_quantile,
            Some(&super::rolling_kernels::no_nulls::rolling_quantile),
        )
    }

    fn rolling_var(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        rolling_agg(
            &self.0,
            options,
            &rolling::no_nulls::rolling_var,
            &rolling::nulls::rolling_var,
            Some(&super::rolling_kernels::no_nulls::rolling_var),
        )
    }

    /// Apply a rolling std (moving std) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their std.
    fn rolling_std(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        rolling_agg(
            &self.0,
            options,
            &rolling::no_nulls::rolling_var,
            &rolling::nulls::rolling_var,
            Some(&super::rolling_kernels::no_nulls::rolling_var),
        )
        .map(|mut s| {
            match s.dtype().clone() {
                DataType::Float32 => {
                    let ca: &mut ChunkedArray<Float32Type> = s._get_inner_mut().as_mut();
                    ca.apply_mut(|v| v.powf(0.5))
                },
                DataType::Float64 => {
                    let ca: &mut ChunkedArray<Float64Type> = s._get_inner_mut().as_mut();
                    ca.apply_mut(|v| v.powf(0.5))
                },
                _ => unreachable!(),
            }
            s
        })
    }
}
