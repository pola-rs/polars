use super::*;
use num::{pow::Pow, Float};
use polars_core::export::num;

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
    fn rolling_mean(&self, options: RollingOptions) -> Result<Series> {
        rolling_agg(
            &self.0,
            options.into(),
            rolling::no_nulls::rolling_mean,
            rolling::nulls::rolling_mean,
        )
    }

    /// Apply a rolling sum (moving sum) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their sum.
    fn rolling_sum(&self, options: RollingOptions) -> Result<Series> {
        rolling_agg(
            &self.0,
            options.into(),
            rolling::no_nulls::rolling_sum,
            rolling::nulls::rolling_sum,
        )
    }

    /// Apply a rolling min (moving min) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their min.
    fn rolling_min(&self, options: RollingOptions) -> Result<Series> {
        rolling_agg(
            &self.0,
            options.into(),
            rolling::no_nulls::rolling_min,
            rolling::nulls::rolling_min,
        )
    }

    /// Apply a rolling max (moving max) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their max.
    fn rolling_max(&self, options: RollingOptions) -> Result<Series> {
        rolling_agg(
            &self.0,
            options.into(),
            rolling::no_nulls::rolling_max,
            rolling::nulls::rolling_max,
        )
    }

    /// Apply a rolling median (moving median) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be weighted according to the `weights` vector.
    fn rolling_median(&self, options: RollingOptions) -> Result<Series> {
        rolling_agg(
            &self.0,
            options.into(),
            rolling::no_nulls::rolling_median,
            rolling::nulls::rolling_median,
        )
    }

    /// Apply a rolling quantile (moving quantile) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be weighted according to the `weights` vector.
    fn rolling_quantile(
        &self,
        quantile: f64,
        interpolation: QuantileInterpolOptions,
        options: RollingOptions,
    ) -> Result<Series> {
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

    fn rolling_var(&self, options: RollingOptions) -> Result<Series> {
        let options_fixed: RollingOptionsFixedWindow = options.clone().into();
        check_input(options_fixed.window_size, options.min_periods)?;
        let ca = self.0.rechunk();

        let arr = ca.downcast_iter().next().unwrap();
        let arr = match self.0.has_validity() {
            false => rolling::no_nulls::rolling_var(
                arr.values(),
                options_fixed.window_size,
                options_fixed.min_periods,
                options_fixed.center,
                options_fixed.weights.as_deref(),
            ),
            _ => rolling::nulls::rolling_var(
                arr,
                options_fixed.window_size,
                options_fixed.min_periods,
                options_fixed.center,
                options_fixed.weights.as_deref(),
            ),
        };
        Series::try_from((self.0.name(), arr))
    }

    /// Apply a rolling std (moving std) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their std.
    fn rolling_std(&self, options: RollingOptions) -> Result<Series> {
        let options_fixed: RollingOptionsFixedWindow = options.clone().into();
        check_input(options_fixed.window_size, options.min_periods)?;
        let ca = self.0.rechunk();

        // weights is only implemented by var kernel
        if options.weights.is_some() {
            return ca
                .into_series()
                .rolling_var(options)
                .and_then(|ca| ca.pow(0.5));
        }
        let options: RollingOptionsFixedWindow = options.into();

        let arr = ca.downcast_iter().next().unwrap();
        let arr = match self.0.has_validity() {
            false => rolling::no_nulls::rolling_std(
                arr.values(),
                options.window_size,
                options.min_periods,
                options.center,
                options.weights.as_deref(),
            ),
            _ => rolling::nulls::rolling_std(
                arr,
                options.window_size,
                options.min_periods,
                options.center,
                options.weights.as_deref(),
            ),
        };
        Series::try_from((self.0.name(), arr))
    }
}
