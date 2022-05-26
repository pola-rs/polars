use super::*;
use std::ops::Deref;

impl SeriesOpsTime for Wrap<DateChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
