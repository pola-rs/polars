use std::fmt::Write;
#[cfg(feature = "timezones")]
use std::str::FromStr;

#[cfg(feature = "timezones")]
use arrow::legacy::kernels::convert_to_naive_local;
use arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};
#[cfg(feature = "timezones")]
use chrono::TimeZone as TimeZoneTrait;
#[cfg(feature = "timezones")]
use chrono_tz::UTC;

use super::*;
#[cfg(feature = "timezones")]
use crate::chunked_array::ops::arity::try_binary_elementwise;
use crate::prelude::DataType::Datetime;
use crate::prelude::*;

impl DatetimeChunked {
    pub fn as_datetime_iter(&self) -> impl TrustedLen<Item = Option<NaiveDateTime>> + '_ {
        let func = match self.time_unit() {
            TimeUnit::Nanoseconds => timestamp_ns_to_datetime,
            TimeUnit::Microseconds => timestamp_us_to_datetime,
            TimeUnit::Milliseconds => timestamp_ms_to_datetime,
        };
        // we know the iterators len
        unsafe {
            self.physical()
                .downcast_iter()
                .flat_map(move |iter| iter.into_iter().map(move |opt_v| opt_v.copied().map(func)))
                .trust_my_length(self.len())
        }
    }

    pub fn time_unit(&self) -> TimeUnit {
        match &self.dtype {
            DataType::Datetime(tu, _) => *tu,
            _ => unreachable!(),
        }
    }

    pub fn time_zone(&self) -> &Option<TimeZone> {
        match &self.dtype {
            DataType::Datetime(_, tz) => tz,
            _ => unreachable!(),
        }
    }

    pub fn time_zone_arc(&self) -> Option<Arc<TimeZone>> {
        match &self.dtype {
            DataType::Datetime(_, tz) => tz.as_ref().map(|tz| Arc::new(tz.clone())),
            _ => unreachable!(),
        }
    }

    /// Convert from Datetime into String with the given format.
    /// See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    pub fn to_string(&self, format: &str) -> PolarsResult<StringChunked> {
        let conversion_f = match self.time_unit() {
            TimeUnit::Nanoseconds => timestamp_ns_to_datetime,
            TimeUnit::Microseconds => timestamp_us_to_datetime,
            TimeUnit::Milliseconds => timestamp_ms_to_datetime,
        };
        let format = get_strftime_format(format, self.dtype())?;
        let mut ca: StringChunked = match self.time_zone() {
            #[cfg(feature = "timezones")]
            Some(time_zone) => {
                let parsed_time_zone = time_zone.parse::<Tz>().expect("already validated");
                let datefmt_f = |ndt| parsed_time_zone.from_utc_datetime(&ndt).format(&format);
                self.physical().try_apply_into_string_amortized(|val, buf| {
                    let ndt = conversion_f(val);
                    write!(buf, "{}", datefmt_f(ndt))
                    }
                ).map_err(
                |_| polars_err!(ComputeError: "cannot format timezone-aware Datetime with format '{}'", format),
                )?
            },
            _ => {
                let datefmt_f = |ndt: NaiveDateTime| ndt.format(&format);
                self.physical().try_apply_into_string_amortized(|val, buf| {
                    let ndt = conversion_f(val);
                    write!(buf, "{}", datefmt_f(ndt))
                    }
                ).map_err(
                |_| polars_err!(ComputeError: "cannot format timezone-naive Datetime with format '{}'", format),
                )?
            },
        };
        ca.rename(self.name().clone());
        Ok(ca)
    }

    /// Convert from Datetime into String with the given format.
    /// See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    ///
    /// Alias for `to_string`.
    pub fn strftime(&self, format: &str) -> PolarsResult<StringChunked> {
        self.to_string(format)
    }

    /// Construct a new [`DatetimeChunked`] from an iterator over [`NaiveDateTime`].
    pub fn from_naive_datetime<I: IntoIterator<Item = NaiveDateTime>>(
        name: PlSmallStr,
        v: I,
        tu: TimeUnit,
    ) -> Self {
        let func = match tu {
            TimeUnit::Nanoseconds => datetime_to_timestamp_ns,
            TimeUnit::Microseconds => datetime_to_timestamp_us,
            TimeUnit::Milliseconds => datetime_to_timestamp_ms,
        };
        let vals = v.into_iter().map(func).collect::<Vec<_>>();
        Int64Chunked::from_vec(name, vals).into_datetime(tu, None)
    }

    pub fn from_naive_datetime_options<I: IntoIterator<Item = Option<NaiveDateTime>>>(
        name: PlSmallStr,
        v: I,
        tu: TimeUnit,
    ) -> Self {
        let func = match tu {
            TimeUnit::Nanoseconds => datetime_to_timestamp_ns,
            TimeUnit::Microseconds => datetime_to_timestamp_us,
            TimeUnit::Milliseconds => datetime_to_timestamp_ms,
        };
        let vals = v.into_iter().map(|opt_nd| opt_nd.map(func));
        Int64Chunked::from_iter_options(name, vals).into_datetime(tu, None)
    }

    /// Change the underlying [`TimeUnit`]. And update the data accordingly.
    #[must_use]
    pub fn cast_time_unit(&self, tu: TimeUnit) -> Self {
        let current_unit = self.time_unit();
        let mut out = self.clone();
        out.set_time_unit(tu);

        use crate::datatypes::time_unit::TimeUnit::*;
        match (current_unit, tu) {
            (Nanoseconds, Microseconds) => {
                let ca = (&self.phys).wrapping_floor_div_scalar(1_000);
                out.phys = ca;
                out
            },
            (Nanoseconds, Milliseconds) => {
                let ca = (&self.phys).wrapping_floor_div_scalar(1_000_000);
                out.phys = ca;
                out
            },
            (Microseconds, Nanoseconds) => {
                let ca = &self.phys * 1_000;
                out.phys = ca;
                out
            },
            (Microseconds, Milliseconds) => {
                let ca = (&self.phys).wrapping_floor_div_scalar(1_000);
                out.phys = ca;
                out
            },
            (Milliseconds, Nanoseconds) => {
                let ca = &self.phys * 1_000_000;
                out.phys = ca;
                out
            },
            (Milliseconds, Microseconds) => {
                let ca = &self.phys * 1_000;
                out.phys = ca;
                out
            },
            (Nanoseconds, Nanoseconds)
            | (Microseconds, Microseconds)
            | (Milliseconds, Milliseconds) => out,
        }
    }

    /// Change the underlying [`TimeUnit`]. This does not modify the data.
    pub fn set_time_unit(&mut self, time_unit: TimeUnit) {
        self.dtype = Datetime(time_unit, self.time_zone().clone());
    }

    /// Change the underlying [`TimeZone`]. This does not modify the data.
    /// This does not validate the time zone - it's up to the caller to verify that it's
    /// already been validated.
    #[cfg(feature = "timezones")]
    pub fn set_time_zone(&mut self, time_zone: TimeZone) -> PolarsResult<()> {
        self.dtype = Datetime(self.time_unit(), Some(time_zone));
        Ok(())
    }

    /// Change the underlying [`TimeUnit`] and [`TimeZone`]. This does not modify the data.
    /// This does not validate the time zone - it's up to the caller to verify that it's
    /// already been validated.
    #[cfg(feature = "timezones")]
    pub fn set_time_unit_and_time_zone(
        &mut self,
        time_unit: TimeUnit,
        time_zone: TimeZone,
    ) -> PolarsResult<()> {
        self.dtype = Datetime(time_unit, Some(time_zone));
        Ok(())
    }

    /// Replace the underlying [`TimeZone`], maintaining the time stamp. This modifies the data.
    #[cfg(feature = "timezones")]
    pub fn replace_time_zone(
        &self,
        time_zone: Option<&TimeZone>,
        ambiguous: &StringChunked,
        non_existent: NonExistent,
    ) -> PolarsResult<DatetimeChunked> {
        let from_time_zone = self.time_zone().clone().unwrap_or(TimeZone::UTC);

        let from_tz = from_time_zone.to_chrono()?;

        let to_tz = if let Some(tz) = time_zone {
            tz.to_chrono()?
        } else {
            chrono_tz::UTC
        };

        if (from_tz == to_tz)
            & ((from_tz == UTC) | ((ambiguous.len() == 1) & (ambiguous.get(0) == Some("raise"))))
        {
            let mut out = self
                .phys
                .clone()
                .into_datetime(self.time_unit(), time_zone.cloned());
            out.physical_mut()
                .set_sorted_flag(self.physical().is_sorted_flag());
            return Ok(out);
        }
        let timestamp_to_datetime: fn(i64) -> NaiveDateTime = match self.time_unit() {
            TimeUnit::Milliseconds => timestamp_ms_to_datetime,
            TimeUnit::Microseconds => timestamp_us_to_datetime,
            TimeUnit::Nanoseconds => timestamp_ns_to_datetime,
        };
        let datetime_to_timestamp: fn(NaiveDateTime) -> i64 = match self.time_unit() {
            TimeUnit::Milliseconds => datetime_to_timestamp_ms,
            TimeUnit::Microseconds => datetime_to_timestamp_us,
            TimeUnit::Nanoseconds => datetime_to_timestamp_ns,
        };

        let out = if ambiguous.len() == 1
            && ambiguous.get(0) != Some("null")
            && non_existent == NonExistent::Raise
        {
            impl_replace_time_zone_fast(
                self,
                ambiguous.get(0),
                timestamp_to_datetime,
                datetime_to_timestamp,
                &from_tz,
                &to_tz,
            )
        } else {
            impl_replace_time_zone(
                self,
                ambiguous,
                non_existent,
                timestamp_to_datetime,
                datetime_to_timestamp,
                &from_tz,
                &to_tz,
            )
        };

        let mut out = out?.into_datetime(self.time_unit(), time_zone.cloned());
        if from_time_zone == TimeZone::UTC
            && ambiguous.len() == 1
            && ambiguous.get(0) == Some("raise")
        {
            // In general, the sortedness flag can't be preserved.
            // To be safe, we only do so in the simplest case when we know for sure that there is no "daylight savings weirdness" going on, i.e.:
            // - `from_tz` is guaranteed to not observe daylight savings time;
            // - user is just passing 'raise' to 'ambiguous'.
            // Both conditions above need to be satisfied.
            out.physical_mut()
                .set_sorted_flag(self.physical().is_sorted_flag());
        }
        Ok(out)
    }
}

#[cfg(feature = "timezones")]
/// If `ambiguous` is length-1 and not equal to "null", we can take a slightly faster path.
fn impl_replace_time_zone_fast(
    datetime: &Logical<DatetimeType, Int64Type>,
    ambiguous: Option<&str>,
    timestamp_to_datetime: fn(i64) -> NaiveDateTime,
    datetime_to_timestamp: fn(NaiveDateTime) -> i64,
    from_tz: &chrono_tz::Tz,
    to_tz: &chrono_tz::Tz,
) -> PolarsResult<Int64Chunked> {
    match ambiguous {
        Some(ambiguous) => datetime.phys.try_apply_nonnull_values_generic(|timestamp| {
            let ndt = timestamp_to_datetime(timestamp);
            Ok(datetime_to_timestamp(
                convert_to_naive_local(
                    from_tz,
                    to_tz,
                    ndt,
                    Ambiguous::from_str(ambiguous)?,
                    NonExistent::Raise,
                )?
                .expect("we didn't use Ambiguous::Null or NonExistent::Null"),
            ))
        }),
        _ => Ok(datetime.phys.apply(|_| None)),
    }
}

#[cfg(feature = "timezones")]
fn impl_replace_time_zone(
    datetime: &Logical<DatetimeType, Int64Type>,
    ambiguous: &StringChunked,
    non_existent: NonExistent,
    timestamp_to_datetime: fn(i64) -> NaiveDateTime,
    datetime_to_timestamp: fn(NaiveDateTime) -> i64,
    from_tz: &chrono_tz::Tz,
    to_tz: &chrono_tz::Tz,
) -> PolarsResult<Int64Chunked> {
    match ambiguous.len() {
        1 => {
            let iter = datetime.phys.downcast_iter().map(|arr| {
                let element_iter = arr.iter().map(|timestamp_opt| match timestamp_opt {
                    Some(timestamp) => {
                        let ndt = timestamp_to_datetime(*timestamp);
                        let res = convert_to_naive_local(
                            from_tz,
                            to_tz,
                            ndt,
                            Ambiguous::from_str(ambiguous.get(0).unwrap())?,
                            non_existent,
                        )?;
                        Ok::<_, PolarsError>(res.map(datetime_to_timestamp))
                    },
                    None => Ok(None),
                });
                element_iter.try_collect_arr()
            });
            ChunkedArray::try_from_chunk_iter(datetime.phys.name().clone(), iter)
        },
        _ => try_binary_elementwise(
            datetime.physical(),
            ambiguous,
            |timestamp_opt, ambiguous_opt| match (timestamp_opt, ambiguous_opt) {
                (Some(timestamp), Some(ambiguous)) => {
                    let ndt = timestamp_to_datetime(timestamp);
                    Ok(convert_to_naive_local(
                        from_tz,
                        to_tz,
                        ndt,
                        Ambiguous::from_str(ambiguous)?,
                        non_existent,
                    )?
                    .map(datetime_to_timestamp))
                },
                _ => Ok(None),
            },
        ),
    }
}

#[cfg(test)]
mod test {
    use chrono::NaiveDateTime;

    use crate::prelude::*;

    #[test]
    fn from_datetime() {
        let datetimes: Vec<_> = [
            "1988-08-25 00:00:16",
            "2015-09-05 23:56:04",
            "2012-12-21 00:00:00",
        ]
        .iter()
        .map(|s| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").unwrap())
        .collect();

        // NOTE: the values are checked and correct.
        let dt = DatetimeChunked::from_naive_datetime(
            PlSmallStr::from_static("name"),
            datetimes.iter().copied(),
            TimeUnit::Nanoseconds,
        );
        assert_eq!(
            [
                588_470_416_000_000_000,
                1_441_497_364_000_000_000,
                1_356_048_000_000_000_000
            ],
            dt.physical().cont_slice().unwrap()
        );
    }
}
