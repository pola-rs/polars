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
        if options.by.is_some() {
            panic!("'rolling by' not yet supported for 'rolling_median', consider using 'groupby_rolling'")
        }
        rolling_agg(
            &self.0,
            options,
            &rolling::no_nulls::rolling_median,
            &rolling::nulls::rolling_median,
            None,
        )
    }

    /// Apply a rolling quantile (moving quantile) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be weighted according to the `weights` vector.
    fn rolling_quantile(
        &self,
        quantile: f64,
        interpolation: QuantileInterpolOptions,
        options: RollingOptionsImpl,
    ) -> PolarsResult<Series> {
        if options.by.is_some() {
            panic!("'rolling by' not yet supported for 'rolling_quantile', consider using 'groupby_rolling'")
        }

        let options: RollingOptionsFixedWindow = options.into();
        check_input(options.window_size, options.min_periods)?;
        let ca = self.0.rechunk();

        let arr = ca.downcast_iter().next().unwrap();
        let arr = match self.0.has_validity() {
            false => rolling::no_nulls::rolling_quantile(
                arr.values(),
                quantile,
                interpolation,
                options.window_size,
                options.min_periods,
                options.center,
                options.weights.as_deref(),
            ),
            _ => rolling::nulls::rolling_quantile(
                arr,
                quantile,
                interpolation,
                options.window_size,
                options.min_periods,
                options.center,
                options.weights.as_deref(),
            ),
        };
        Series::try_from((self.0.name(), arr))
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
        if options.window_size.parsed_int {
            let options_fixed: RollingOptionsFixedWindow = options.clone().into();
            check_input(options_fixed.window_size, options.min_periods)?;
        }

        // weights is only implemented by var kernel
        if options.weights.is_some() {
            return self
                .0
                .clone()
                .into_series()
                .rolling_var(options)
                .map(|mut s| {
                    match s.dtype().clone() {
                        DataType::Float32 => {
                            let ca: &mut ChunkedArray<Float32Type> = s._get_inner_mut().as_mut();
                            ca.apply_mut(|v| v.powf(0.5))
                        }
                        DataType::Float64 => {
                            let ca: &mut ChunkedArray<Float64Type> = s._get_inner_mut().as_mut();
                            ca.apply_mut(|v| v.powf(0.5))
                        }
                        _ => unreachable!(),
                    }
                    s
                });
        }

        rolling_agg(
            &self.0,
            options,
            &rolling::no_nulls::rolling_std,
            &rolling::nulls::rolling_std,
            Some(&super::rolling_kernels::no_nulls::rolling_std),
        )
    }
}
