use super::*;
use std::ops::Deref;

impl SeriesOps for Wrap<DurationChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
