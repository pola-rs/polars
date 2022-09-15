use super::*;
use crate::prelude::*;

pub type TimeChunked = Logical<TimeType, Int64Type>;

impl From<Int64Chunked> for TimeChunked {
    fn from(ca: Int64Chunked) -> Self {
        TimeChunked::new_logical(ca)
    }
}

impl Int64Chunked {
    pub fn into_time(self) -> TimeChunked {
        TimeChunked::new_logical(self)
    }
}

impl LogicalType for TimeChunked {
    fn dtype(&self) -> &'static DataType {
        &DataType::Time
    }

    #[cfg(feature = "dtype-time")]
    fn get_any_value(&self, i: usize) -> AnyValue<'_> {
        self.0.get_any_value(i).into_time()
    }

    fn cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        self.0.cast(dtype)
    }
}
