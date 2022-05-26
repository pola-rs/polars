use super::*;

impl SeriesOpsTime for Wrap<StructChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
