use super::*;
#[cfg(feature = "to_dummies")]
use std::ops::Deref;

impl SeriesOps for Wrap<DatetimeChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
