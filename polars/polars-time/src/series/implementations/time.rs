use super::*;
use std::ops::Deref;

impl SeriesOpsTime for Wrap<TimeChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
