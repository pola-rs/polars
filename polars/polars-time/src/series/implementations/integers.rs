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

}
