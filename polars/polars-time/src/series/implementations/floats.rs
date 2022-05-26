use std::ops::SubAssign;
use polars_arrow::data_types::IsFloat;
use super::*;

impl<T: PolarsFloatType> SeriesOps for WrapFloat<ChunkedArray<T>>
where T::Native: IsFloat + SubAssign
{
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }

    #[cfg(feature = "rolling_window")]
    fn rolling_mean(&self, _options: RollingOptions) -> Result<Series> {
        self.rolling_mean(_options)
    }
    // #[cfg(feature = "rolling_window")]
    // fn rolling_sum(&self, _options: RollingOptions) -> Result<Series> {
    //     self.to_ops().rolling_sum(_options)
    // }
    // #[cfg(feature = "rolling_window")]
    // fn rolling_median(&self, _options: RollingOptions) -> Result<Series> {
    //     self.to_ops().rolling_median(_options)
    // }
    // /// Apply a rolling quantile to a Series.
    // #[cfg(feature = "rolling_window")]
    // fn rolling_quantile(
    //     &self,
    //     quantile: f64,
    //     interpolation: QuantileInterpolOptions,
    //     options: RollingOptions,
    // ) -> Result<Series> {
    //     self.to_ops().rolling_quantile(
    //         quantile,
    //         interpolation,
    //         options
    //     )
    // }
    //
    // #[cfg(feature = "rolling_window")]
    // fn rolling_min(&self, options: RollingOptions) -> Result<Series> {
    //     self.to_ops().rolling_min(options)
    // }
    // /// Apply a rolling max to a Series.
    // #[cfg(feature = "rolling_window")]
    // fn rolling_max(&self,options: RollingOptions) -> Result<Series> {
    //     self.to_ops().rolling_max(options)
    // }
    //
    // /// Apply a rolling variance to a Series.
    // #[cfg(feature = "rolling_window")]
    // fn rolling_var(&self, options: RollingOptions) -> Result<Series> {
    //     self.to_ops().rolling_var(options)
    // }
    //
    // /// Apply a rolling std_dev to a Series.
    // #[cfg(feature = "rolling_window")]
    // fn rolling_std(&self, options: RollingOptions) -> Result<Series> {
    //     self.to_ops().rolling_std(options)
    // }
}
