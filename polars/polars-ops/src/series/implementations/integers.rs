use super::*;

impl<T: PolarsIntegerType> SeriesOps for WrapInt<ChunkedArray<T>>
where
    T::Native: NumericNative,
{
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
    #[cfg(feature = "to_dummies")]
    fn to_dummies(&self) -> PolarsResult<DataFrame> {
        ToDummies::to_dummies(&self.0)
    }
}
