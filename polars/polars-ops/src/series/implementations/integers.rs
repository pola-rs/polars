use super::*;

impl<T: PolarsIntegerType> SeriesOps for WrapInt<ChunkedArray<T>>
where
    T::Native: NumericNative,
{
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
