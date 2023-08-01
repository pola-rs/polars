use super::*;
use crate::prelude::*;

pub type DatetimeChunked = Logical<DatetimeType, Int64Type>;

impl Int64Chunked {
    pub fn into_datetime(self, timeunit: TimeUnit, tz: Option<TimeZone>) -> DatetimeChunked {
        let mut dt = DatetimeChunked::new_logical(self);
        dt.2 = Some(DataType::Datetime(timeunit, tz));
        dt
    }
}

impl LogicalType for DatetimeChunked {
    fn dtype(&self) -> &DataType {
        self.2.as_ref().unwrap()
    }

    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        self.0
            .get_any_value(i)
            .map(|av| av.into_datetime(self.time_unit(), self.time_zone()))
    }

    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        self.0
            .get_any_value_unchecked(i)
            .into_datetime(self.time_unit(), self.time_zone())
    }

    fn cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        use DataType::*;
        match (self.dtype(), dtype) {
            (Datetime(TimeUnit::Milliseconds, _), Datetime(TimeUnit::Nanoseconds, tz)) => {
                Ok((self.0.as_ref() * 1_000_000i64)
                    .into_datetime(TimeUnit::Nanoseconds, tz.clone())
                    .into_series())
            }
            (Datetime(TimeUnit::Milliseconds, _), Datetime(TimeUnit::Microseconds, tz)) => {
                Ok((self.0.as_ref() * 1_000i64)
                    .into_datetime(TimeUnit::Microseconds, tz.clone())
                    .into_series())
            }
            (Datetime(TimeUnit::Nanoseconds, _), Datetime(TimeUnit::Milliseconds, tz)) => {
                Ok((self.0.as_ref() / 1_000_000i64)
                    .into_datetime(TimeUnit::Milliseconds, tz.clone())
                    .into_series())
            }
            (Datetime(TimeUnit::Nanoseconds, _), Datetime(TimeUnit::Microseconds, tz)) => {
                Ok((self.0.as_ref() / 1_000i64)
                    .into_datetime(TimeUnit::Microseconds, tz.clone())
                    .into_series())
            }
            (Datetime(TimeUnit::Microseconds, _), Datetime(TimeUnit::Milliseconds, tz)) => {
                Ok((self.0.as_ref() / 1_000i64)
                    .into_datetime(TimeUnit::Milliseconds, tz.clone())
                    .into_series())
            }
            (Datetime(TimeUnit::Microseconds, _), Datetime(TimeUnit::Nanoseconds, tz)) => {
                Ok((self.0.as_ref() * 1_000i64)
                    .into_datetime(TimeUnit::Nanoseconds, tz.clone())
                    .into_series())
            }
            #[cfg(feature = "dtype-date")]
            (Datetime(tu, _), Date) => match tu {
                TimeUnit::Nanoseconds => Ok((self.0.as_ref() / NS_IN_DAY)
                    .cast(&Int32)
                    .unwrap()
                    .into_date()
                    .into_series()),
                TimeUnit::Microseconds => Ok((self.0.as_ref() / US_IN_DAY)
                    .cast(&Int32)
                    .unwrap()
                    .into_date()
                    .into_series()),
                TimeUnit::Milliseconds => Ok((self.0.as_ref() / MS_IN_DAY)
                    .cast(&Int32)
                    .unwrap()
                    .into_date()
                    .into_series()),
            },
            #[cfg(feature = "dtype-time")]
            (Datetime(tu, _), Time) => match tu {
                TimeUnit::Nanoseconds => Ok((self.0.as_ref() % NS_IN_DAY)
                    .cast(&Int64)
                    .unwrap()
                    .into_time()
                    .into_series()),
                TimeUnit::Microseconds => Ok((self.0.as_ref() % US_IN_DAY * 1_000i64)
                    .cast(&Int64)
                    .unwrap()
                    .into_time()
                    .into_series()),
                TimeUnit::Milliseconds => Ok((self.0.as_ref() % MS_IN_DAY * 1_000_000i64)
                    .cast(&Int64)
                    .unwrap()
                    .into_time()
                    .into_series()),
            },
            _ => self.0.cast(dtype),
        }
    }
}
