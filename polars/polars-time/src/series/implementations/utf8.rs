use super::*;

impl SeriesOps for Wrap<Utf8Chunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
