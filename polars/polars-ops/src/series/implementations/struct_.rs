use super::*;

impl SeriesOps for Wrap<StructChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
