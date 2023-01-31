use super::*;
use crate::prelude::*;
pub type DateChunked = Logical<DateType, Int32Type>;

impl From<Int32Chunked> for DateChunked {
    fn from(ca: Int32Chunked) -> Self {
        DateChunked::new_logical(ca)
    }
}

impl Int32Chunked {
    pub fn into_date(self) -> DateChunked {
        DateChunked::new_logical(self)
    }
}

impl LogicalType for DateChunked {
    fn dtype(&self) -> &DataType {
        &DataType::Date
    }

    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        self.0.get_any_value(i).map(|av| av.into_date())
    }

    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        self.0.get_any_value_unchecked(i).into_date()
    }

    fn cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        use DataType::*;
        match (self.dtype(), dtype) {
            #[cfg(feature = "dtype-datetime")]
            (Date, Datetime(tu, tz)) => {
                let casted = self.0.cast(dtype)?;
                let casted = casted.datetime().unwrap();
                let conversion = match tu {
                    TimeUnit::Nanoseconds => NS_IN_DAY,
                    TimeUnit::Microseconds => US_IN_DAY,
                    TimeUnit::Milliseconds => MS_IN_DAY,
                };
                Ok((casted.deref() * conversion)
                    .into_datetime(*tu, tz.clone())
                    .into_series())
            }
            (Date, Time) => Ok(Int64Chunked::full(self.name(), 0i64, self.len())
                .into_time()
                .into_series()),
            _ => self.0.cast(dtype),
        }
    }
}
