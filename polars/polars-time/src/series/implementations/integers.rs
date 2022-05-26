use polars_arrow::data_types::IsFloat;
use super::*;

impl<T: PolarsIntegerType> SeriesOps for WrapInt<ChunkedArray<T>>
where
    T::Native: NumericNative ,
Self: RollingAgg
{
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }

    #[cfg(feature = "rolling_window")]
    fn rolling_mean(&self, options: RollingOptions) -> Result<Series> {
        let s = self.0.cast(&DataType::Float64).unwrap();
        s.rolling_mean(options)
    }

    #[cfg(feature = "rolling_window")]
    fn rolling_sum(&self, options: RollingOptions) -> Result<Series> {
        RollingAgg::rolling_sum(self, options)
    }
    #[cfg(feature = "rolling_window")]
    fn rolling_median(&self, options: RollingOptions) -> Result<Series> {
        RollingAgg::rolling_sum(self, options)
    }
    /// Apply a rolling quantile to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_quantile(
        &self,
        quantile: f64,
        interpolation: QuantileInterpolOptions,
        options: RollingOptions,
    ) -> Result<Series> {
        RollingAgg::rolling_quantile(self,
                                     quantile,
                                     interpolation,
                                     options
        )
    }
    //
    #[cfg(feature = "rolling_window")]
    fn rolling_min(&self, options: RollingOptions) -> Result<Series> {
        RollingAgg::rolling_min(self, options)
    }

    #[cfg(feature = "rolling_window")]
    fn rolling_max(&self,options: RollingOptions) -> Result<Series> {
        RollingAgg::rolling_max(self, options)
    }
    #[cfg(feature = "rolling_window")]
    fn rolling_var(&self, options: RollingOptions) -> Result<Series> {
        let s = self.0.cast(&DataType::Float64).unwrap();
        s.rolling_var(options)
    }

    /// Apply a rolling std_dev to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_std(&self, options: RollingOptions) -> Result<Series> {
        let s = self.0.cast(&DataType::Float64).unwrap();
        s.rolling_std(options)
    }

}
