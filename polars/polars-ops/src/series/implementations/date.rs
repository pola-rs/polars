use super::*;

impl SeriesOps for Wrap<DateChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
