use super::*;
use crate::series::WrapInt;

#[cfg(not(feature = "rolling_window"))]
impl<T> RollingAgg for WrapInt<ChunkedArray<T>>
where
    T: PolarsIntegerType,
    T::Native: IsFloat + SubAssign,
{
}

#[cfg(feature = "rolling_window")]
impl<T> RollingAgg for WrapInt<ChunkedArray<T>>
where
    T: PolarsIntegerType,
    T::Native: IsFloat + SubAssign,
{
    fn rolling_sum(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        if options.weights.is_some() {
            return self.0.cast(&DataType::Float64)?.rolling_sum(options);
        }
        rolling_agg(
            &self.0,
            options,
            &rolling::no_nulls::rolling_sum,
            &rolling::nulls::rolling_sum,
            Some(&super::rolling_kernels::no_nulls::rolling_sum),
        )
    }

    fn rolling_median(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        self.0.cast(&DataType::Float64)?.rolling_median(options)
    }

    fn rolling_quantile(
        &self,
        quantile: f64,
        interpolation: QuantileInterpolOptions,
        options: RollingOptionsImpl,
    ) -> PolarsResult<Series> {
        self.0
            .cast(&DataType::Float64)?
            .rolling_quantile(quantile, interpolation, options)
    }

    fn rolling_min(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        if options.weights.is_some() {
            return self.0.cast(&DataType::Float64)?.rolling_min(options);
        }
        rolling_agg(
            &self.0,
            options,
            &rolling::no_nulls::rolling_min,
            &rolling::nulls::rolling_min,
            Some(&super::rolling_kernels::no_nulls::rolling_min),
        )
    }

    fn rolling_max(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        if options.weights.is_some() {
            return self.0.cast(&DataType::Float64)?.rolling_max(options);
        }
        rolling_agg(
            &self.0,
            options,
            &rolling::no_nulls::rolling_max,
            &rolling::nulls::rolling_max,
            Some(&super::rolling_kernels::no_nulls::rolling_max),
        )
    }

    fn rolling_var(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        self.0.cast(&DataType::Float64)?.rolling_var(options)
    }

    fn rolling_std(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        self.0.cast(&DataType::Float64)?.rolling_std(options)
    }

    fn rolling_mean(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        self.0.cast(&DataType::Float64)?.rolling_mean(options)
    }
}
