use super::*;

impl SeriesOpsTime for Wrap<BooleanChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
