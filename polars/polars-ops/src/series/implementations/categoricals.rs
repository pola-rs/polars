use super::*;

impl SeriesOps for Wrap<CategoricalChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
    #[cfg(feature = "to_dummies")]
    fn to_dummies(&self) -> Result<DataFrame> {
        ToDummies::to_dummies(self)
    }
}
