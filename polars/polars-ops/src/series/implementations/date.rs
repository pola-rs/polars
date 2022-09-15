use std::ops::Deref;

use super::*;

impl SeriesOps for Wrap<DateChunked> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
    #[cfg(feature = "to_dummies")]
    fn to_dummies(&self) -> PolarsResult<DataFrame> {
        ToDummies::to_dummies(self.0.deref())
    }
}
