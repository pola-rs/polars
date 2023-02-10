use super::*;

impl SeriesOps for Wrap<CategoricalChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
