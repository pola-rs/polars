use super::*;
use crate::prelude::*;

pub type DecimalChunked = Logical<DatetimeType, Int128Type>;

impl LogicalType for DecimalChunked {
    fn dtype(&self) -> &DataType {
        self.2.as_ref().unwrap()
    }

    fn cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        todo!()
    }
}
