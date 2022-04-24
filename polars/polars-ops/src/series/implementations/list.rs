use super::*;

impl SeriesOps for Wrap<ListChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
