use super::*;
use crate::prelude::*;

pub type Date32Chunked = Logical<Date32Type, Int32Type>;

impl From<Int32Chunked> for Date32Chunked {
    fn from(ca: Int32Chunked) -> Self {
        Date32Chunked::new(ca)
    }
}

impl Int32Chunked {
    pub fn into_date(self) -> Date32Chunked {
        Date32Chunked::new(self)
    }
}

impl LogicalType for Date32Chunked {
    fn dtype(&self) -> &'static DataType {
        &DataType::Date32
    }
}
