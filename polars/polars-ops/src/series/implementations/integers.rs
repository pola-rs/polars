use super::*;

impl<T: PolarsIntegerType> SeriesOps for WrapInt<ChunkedArray<T>>
where
    T::Native: NumericNative,
    ChunkedArray<T>: ChunkQuantile<f64>,
{
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
    #[cfg(feature = "to_dummies")]
    fn to_dummies(&self) -> Result<DataFrame> {
        ToDummies::to_dummies(&self.0)
    }
    #[cfg(feature = "cut_qcut")]
    fn qcut(&self, bins: Vec<f64>) -> Result<Series> {
        CutQCut::qcut(&self.0, bins)
    }
}
