use super::*;

impl SeriesOps for Wrap<BooleanChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
