use crate::series::WrapInt;
use super::*;


pub trait PolarsTimeIntegerType: PolarsIntegerType {}
impl PolarsTimeIntegerType for UInt8Type {}
impl PolarsTimeIntegerType for UInt16Type {}
impl PolarsTimeIntegerType for UInt32Type {}
impl PolarsTimeIntegerType for UInt64Type {}
impl PolarsTimeIntegerType for Int8Type {}
impl PolarsTimeIntegerType for Int16Type {}
impl PolarsTimeIntegerType for Int32Type {}
impl PolarsTimeIntegerType for Int64Type {}


impl<T> RollingAgg for WrapInt<ChunkedArray<T>>
    where
        T: PolarsIntegerType,
        T::Native: IsFloat + SubAssign,
{
    fn rolling_sum(&self, options: RollingOptions) -> Result<Series> {
        if options.weights.is_some() {
            return self.0.cast(&DataType::Float64)?.rolling_sum(options);
        }
        rolling_agg(
            &self.0,
            options.into(),
            rolling::no_nulls::rolling_sum,
            rolling::nulls::rolling_sum,
        )
    }

    fn rolling_median(&self, options: RollingOptions) -> Result<Series> {
        self.0.cast(&DataType::Float64)?.rolling_median(options)
    }

    fn rolling_quantile(
        &self,
        quantile: f64,
        interpolation: QuantileInterpolOptions,
        options: RollingOptions,
    ) -> Result<Series> {
        self.0.cast(&DataType::Float64)?
            .rolling_quantile(quantile, interpolation, options.into())
    }

    fn rolling_min(&self, options: RollingOptions) -> Result<Series> {
        if options.weights.is_some() {
            return self.0.cast(&DataType::Float64)?.rolling_min(options);
        }
        rolling_agg(
            &self.0,
            options.into(),
            rolling::no_nulls::rolling_min,
            rolling::nulls::rolling_min,
        )
    }

    fn rolling_max(&self, options: RollingOptions) -> Result<Series> {
        if options.weights.is_some() {
            return self.0.cast(&DataType::Float64)?.rolling_max(options);
        }
        rolling_agg(
            &self.0,
            options.into(),
            rolling::no_nulls::rolling_max,
            rolling::nulls::rolling_max,
        )
    }

    fn rolling_var(&self, options: RollingOptions) -> Result<Series> {
        self.0.cast(&DataType::Float64)?
            .rolling_var(options.into())
    }

    fn rolling_std(&self, options: RollingOptions) -> Result<Series> {
        self.0.cast(&DataType::Float64)?
            .rolling_std(options.into())
    }

    fn rolling_mean(&self, options: RollingOptions) -> Result<Series> {
        self.0.cast(&DataType::Float64)?
            .rolling_mean(options.into())
    }
}


