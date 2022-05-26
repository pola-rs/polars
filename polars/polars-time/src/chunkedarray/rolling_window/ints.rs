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
    /// Apply a rolling sum (moving sum) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their sum.
    fn rolling_sum(&self, options: RollingOptions) -> Result<Series> {
        // if options.weights.is_some() {
        //     return self.0.cast(&DataType::Float64)?.rolling_sum(options);
        // }
        rolling_agg(
            &self.0,
            options.into(),
            rolling::no_nulls::rolling_sum,
            rolling::nulls::rolling_sum,
        )
    }

    /// Apply a rolling median (moving median) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be weighted according to the `weights` vector.
    fn rolling_median(&self, options: RollingOptions) -> Result<Series> {
        todo!()
        // self.cast(&DataType::Float64)?.rolling_median(options)
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
        todo!()
        // self.cast(&DataType::Float64)?
        //     .rolling_quantile(quantile, interpolation, options.into())
    }

    /// Apply a rolling min (moving min) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their min.
    fn rolling_min(&self, options: RollingOptions) -> Result<Series> {
        // if options.weights.is_some() {
        //     return self.cast(&DataType::Float64)?.rolling_min(options);
        // }
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
        // if options.weights.is_some() {
        //     return self.cast(&DataType::Float64)?.rolling_max(options);
        // }
        rolling_agg(
            &self.0,
            options.into(),
            rolling::no_nulls::rolling_max,
            rolling::nulls::rolling_max,
        )
    }
}


