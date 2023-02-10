use super::*;

impl<T: PolarsFloatType> SeriesOps for WrapFloat<ChunkedArray<T>> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
