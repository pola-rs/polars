use super::*;

impl SeriesOpsTime for Wrap<CategoricalChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
