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
    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        self.0.get_any_value(i).map(|av| av.into_time())
    }
    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        self.0.get_any_value_unchecked(i).into_time()
    }

    fn cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        match dtype {
            DataType::Duration(tu) => {
                let out = self.0.cast(&DataType::Duration(TimeUnit::Nanoseconds));
                if !matches!(tu, TimeUnit::Nanoseconds) {
                    out?.cast(dtype)
                } else {
                    out
                }
            }
            _ => self.0.cast(dtype),
        }
    }
}
