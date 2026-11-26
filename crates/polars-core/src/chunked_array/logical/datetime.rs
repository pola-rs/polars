use super::*;
use crate::datatypes::time_unit::TimeUnit;
use crate::prelude::*;

pub type DatetimeChunked = Logical<DatetimeType, Int64Type>;

impl Int64Chunked {
    pub fn into_datetime(self, timeunit: TimeUnit, tz: Option<TimeZone>) -> DatetimeChunked {
        // SAFETY: no invalid states.
        unsafe { DatetimeChunked::new_logical(self, DataType::Datetime(timeunit, tz)) }
    }
}

impl DatetimeChunked {
    /// The days since the epoch, or `None` if a day does not fit in an `i32`.
    #[cfg(feature = "dtype-date")]
    fn to_date_values<const PER_DAY: i64>(&self) -> Option<Int32Chunked> {
        let first = (i32::MIN as i64).checked_mul(PER_DAY).unwrap_or(i64::MIN);
        let last = (i32::MAX as i64 + 1)
            .checked_mul(PER_DAY)
            .map_or(i64::MAX, |v| v - 1);
        let in_range = self.phys.downcast_iter().all(|arr| {
            arr.values()
                .iter()
                .fold(true, |acc, &v| acc & (first <= v) & (v <= last))
        });
        in_range
            .then(|| arity::unary_elementwise_values(&self.phys, |v| v.div_euclid(PER_DAY) as i32))
    }

    /// The nanoseconds since midnight.
    #[cfg(feature = "dtype-time")]
    fn to_time_values<const PER_DAY: i64, const TO_NANOSECONDS: i64>(&self) -> Int64Chunked {
        self.phys
            .apply_values(|v| v.rem_euclid(PER_DAY) * TO_NANOSECONDS)
    }
}

impl LogicalType for DatetimeChunked {
    fn dtype(&self) -> &DataType {
        &self.dtype
    }

    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        self.phys
            .get_any_value(i)
            .map(|av| av.as_datetime(self.time_unit(), self.time_zone().as_ref()))
    }

    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        self.phys
            .get_any_value_unchecked(i)
            .as_datetime(self.time_unit(), self.time_zone().as_ref())
    }

    fn cast_with_options(
        &self,
        dtype: &DataType,
        cast_options: CastOptions,
    ) -> PolarsResult<Series> {
        use DataType::*;

        use crate::datatypes::time_unit::TimeUnit::*;

        let out = match dtype {
            Datetime(to_unit, tz) => Ok(self
                .cast_time_unit(*to_unit)
                .phys
                .into_datetime(*to_unit, tz.clone())
                .into_series()),
            #[cfg(feature = "dtype-date")]
            Date => {
                let date = match self.time_unit() {
                    Nanoseconds => self.to_date_values::<NS_IN_DAY>(),
                    Microseconds => self.to_date_values::<US_IN_DAY>(),
                    Milliseconds => self.to_date_values::<MS_IN_DAY>(),
                };
                match date {
                    Some(date) => Ok(date.into_date().into_series()),
                    None => {
                        let tu_in_day = match self.time_unit() {
                            Nanoseconds => NS_IN_DAY,
                            Microseconds => US_IN_DAY,
                            Milliseconds => MS_IN_DAY,
                        };
                        Ok(self
                            .phys
                            .apply_values(|v| v.div_euclid(tu_in_day))
                            .cast_with_options(&Int32, cast_options)
                            .unwrap()
                            .into_date()
                            .into_series())
                    },
                }
            },
            #[cfg(feature = "dtype-time")]
            Time => {
                let time = match self.time_unit() {
                    Nanoseconds => self.to_time_values::<NS_IN_DAY, 1>(),
                    Microseconds => self.to_time_values::<US_IN_DAY, 1_000>(),
                    Milliseconds => self.to_time_values::<MS_IN_DAY, 1_000_000>(),
                };
                return Ok(time.into_time().into_series());
            },
            dt if dt.is_primitive_numeric() => {
                return self.phys.cast_with_options(dtype, cast_options);
            },
            dt => {
                polars_bail!(
                    InvalidOperation:
                    "casting from {:?} to {:?} not supported",
                    self.dtype(), dt
                )
            },
        };
        out.map(|mut s| {
            // Out-of-range values become null, which may break the order of the nulls.
            if s.null_count() == self.null_count() {
                s.set_sorted_flag(self.physical().is_sorted_flag());
            }
            s
        })
    }
}
