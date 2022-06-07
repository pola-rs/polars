use super::*;

impl<T: PolarsFloatType> SeriesOps for WrapFloat<ChunkedArray<T>> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
    #[cfg(feature = "to_dummies")]
    fn to_dummies(&self) -> Result<DataFrame> {
        ToDummies::to_dummies(self)
    }
    // #[cfg(feature = "cut_qcut")]
    // fn qcut(&self, bins: Vec<f64>) -> Result<Series> {
    //    CutQCut::qcut(&self.0, bins)
    //}
    //#[cfg(feature = "cut_qcut")]
    //fn cut(&self, bins: Vec<f64>) -> Result<Series> {
    //    CutQCut::cut(&self.0, bins)
    //}
}
