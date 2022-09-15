use std::ops::SubAssign;

use polars_arrow::data_types::IsFloat;

use super::*;

impl<T: PolarsFloatType> SeriesOpsTime for WrapFloat<ChunkedArray<T>>
where
    T::Native: IsFloat + SubAssign,
    Self: RollingAgg,
{
    fn ops_time_dtype(&self) -> &DataType {
        self.0.dtype()
    }

    #[cfg(feature = "rolling_window")]
    fn rolling_mean(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        RollingAgg::rolling_mean(self, options)
    }
    #[cfg(feature = "rolling_window")]
    fn rolling_sum(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        RollingAgg::rolling_sum(self, options)
    }
    #[cfg(feature = "rolling_window")]
    fn rolling_median(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        RollingAgg::rolling_median(self, options)
    }

    #[cfg(feature = "rolling_window")]
    fn rolling_quantile(
        &self,
        quantile: f64,
        interpolation: QuantileInterpolOptions,
        options: RollingOptionsImpl,
    ) -> PolarsResult<Series> {
        RollingAgg::rolling_quantile(self, quantile, interpolation, options)
    }

    #[cfg(feature = "rolling_window")]
    fn rolling_min(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        RollingAgg::rolling_min(self, options)
    }

    #[cfg(feature = "rolling_window")]
    fn rolling_max(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        RollingAgg::rolling_max(self, options)
    }
    #[cfg(feature = "rolling_window")]
    fn rolling_var(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        RollingAgg::rolling_var(self, options)
    }

    /// Apply a rolling std_dev to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_std(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        RollingAgg::rolling_std(self, options)
    }
}
