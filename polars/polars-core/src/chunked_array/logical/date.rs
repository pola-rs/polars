use super::*;
use crate::prelude::*;

pub type DateChunked = Logical<DateType, Int32Type>;

impl From<Int32Chunked> for DateChunked {
    fn from(ca: Int32Chunked) -> Self {
        DateChunked::new(ca)
    }
}

impl Int32Chunked {
    pub fn into_date(self) -> DateChunked {
        DateChunked::new(self)
    }
}

impl LogicalType for DateChunked {
    fn dtype(&self) -> &'static DataType {
        &DataType::Date
    }
}
