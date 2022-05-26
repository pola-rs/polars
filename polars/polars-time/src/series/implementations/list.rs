use super::*;

impl SeriesOpsTime for Wrap<ListChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
