use super::*;
use crate::prelude::*;

pub type DatetimeChunked = Logical<DatetimeType, Int64Type>;

impl From<Int64Chunked> for DatetimeChunked {
    fn from(ca: Int64Chunked) -> Self {
        DatetimeChunked::new(ca)
    }
}

impl Int64Chunked {
    pub fn into_date(self) -> DatetimeChunked {
        DatetimeChunked::new(self)
    }
}

impl LogicalType for DatetimeChunked {
    fn dtype(&self) -> &'static DataType {
        &DataType::Datetime
    }
}
