use super::*;
#[cfg(feature = "to_dummies")]
use std::ops::Deref;

impl SeriesOps for Wrap<DatetimeChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
    #[cfg(feature = "to_dummies")]
    fn to_dummies(&self) -> Result<DataFrame> {
        self.0.deref().to_ops().to_dummies()
    }
}
