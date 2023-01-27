use super::*;
use crate::prelude::*;

pub type DurationChunked = Logical<DurationType, Int64Type>;

impl Int64Chunked {
    pub fn into_duration(self, timeunit: TimeUnit) -> DurationChunked {
        let mut dt = DurationChunked::new_logical(self);
        dt.2 = Some(DataType::Duration(timeunit));
        dt
    }
}

impl LogicalType for DurationChunked {
    fn dtype(&self) -> &DataType {
        self.2.as_ref().unwrap()
    }

    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        self.0
            .get_any_value(i)
            .map(|av| av.into_duration(self.time_unit()))
    }
    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        self.0
            .get_any_value_unchecked(i)
            .into_duration(self.time_unit())
    }

    fn cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        use DataType::*;
        match (self.dtype(), dtype) {
            (Duration(TimeUnit::Milliseconds), Duration(TimeUnit::Nanoseconds)) => {
                Ok((self.0.as_ref() * 1_000_000i64)
                    .into_duration(TimeUnit::Nanoseconds)
                    .into_series())
            }
            (Duration(TimeUnit::Milliseconds), Duration(TimeUnit::Microseconds)) => {
                Ok((self.0.as_ref() * 1_000i64)
                    .into_duration(TimeUnit::Microseconds)
                    .into_series())
            }
            (Duration(TimeUnit::Microseconds), Duration(TimeUnit::Milliseconds)) => {
                Ok((self.0.as_ref() / 1_000i64)
                    .into_duration(TimeUnit::Milliseconds)
                    .into_series())
            }
            (Duration(TimeUnit::Microseconds), Duration(TimeUnit::Nanoseconds)) => {
                Ok((self.0.as_ref() * 1_000i64)
                    .into_duration(TimeUnit::Nanoseconds)
                    .into_series())
            }
            (Duration(TimeUnit::Nanoseconds), Duration(TimeUnit::Milliseconds)) => {
                Ok((self.0.as_ref() / 1_000_000i64)
                    .into_duration(TimeUnit::Milliseconds)
                    .into_series())
            }
            (Duration(TimeUnit::Nanoseconds), Duration(TimeUnit::Microseconds)) => {
                Ok((self.0.as_ref() / 1_000i64)
                    .into_duration(TimeUnit::Microseconds)
                    .into_series())
            }
            _ => self.0.cast(dtype),
        }
    }
}
