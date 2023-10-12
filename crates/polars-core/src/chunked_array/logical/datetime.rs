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
            (Datetime(tu_l, _), Datetime(tu_r, tz)) => {
                Ok(convert_time_units(self.0.clone(), tu_l, tu_r)
                    .into_datetime(*tu_r, tz.clone())
                    .into_series())
            },
            #[cfg(feature = "dtype-date")]
            (Datetime(tu, _), Date) => Ok((self.0.as_ref() / get_seconds_in_day(tu))
                .cast(&Int32)
                .unwrap()
                .into_date()
                .into_series()),
            #[cfg(feature = "dtype-time")]
            (Datetime(tu, _), Time) => {
                let v = self.0.as_ref() % get_seconds_in_day(tu);
                let v = match tu {
                    TimeUnit::Nanoseconds => v,
                    TimeUnit::Microseconds => v * 1000i64,
                    TimeUnit::Milliseconds => v * 1_000_000i64,
                };
                Ok(v.cast(&Int64).unwrap().into_time().into_series())
            },
            _ => self.0.cast(dtype),
        }
    }
}
