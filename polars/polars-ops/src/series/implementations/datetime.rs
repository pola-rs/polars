use super::*;

impl SeriesOps for Wrap<DatetimeChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
