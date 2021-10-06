use super::*;
use crate::prelude::*;

pub type Date64Chunked = Logical<Date64Type, Int64Type>;

impl From<Int64Chunked> for Date64Chunked {
    fn from(ca: Int64Chunked) -> Self {
        Date64Chunked::new(ca)
    }
}

impl Int64Chunked {
    pub fn into_date(self) -> Date64Chunked {
        Date64Chunked::new(self)
    }
}

impl LogicalType for Date64Chunked {
    fn dtype(&self) -> &'static DataType {
        &DataType::Date64
    }
}
