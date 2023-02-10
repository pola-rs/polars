use super::*;

impl SeriesOps for Wrap<TimeChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
