use std::fmt::Write;

use arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};
#[cfg(feature = "timezones")]
use chrono::TimeZone as TimeZoneTrait;

use super::*;
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
        match tu {
            TimeUnit::Nanoseconds => {
                let vals = v.into_iter().map(|ndt| ndt.and_utc().timestamp_nanos_opt());
                Int64Chunked::from_iter_options(name, vals).into_datetime(tu, None)
            },
            TimeUnit::Microseconds => {
                let vals = v
                    .into_iter()
                    .map(datetime_to_timestamp_us)
                    .collect::<Vec<_>>();
                Int64Chunked::from_vec(name, vals).into_datetime(tu, None)
            },
            TimeUnit::Milliseconds => {
                let vals = v
                    .into_iter()
                    .map(datetime_to_timestamp_ms)
                    .collect::<Vec<_>>();
                Int64Chunked::from_vec(name, vals).into_datetime(tu, None)
            },
        }
    }

    pub fn from_naive_datetime_options<I: IntoIterator<Item = Option<NaiveDateTime>>>(
        name: PlSmallStr,
        v: I,
        tu: TimeUnit,
    ) -> Self {
        match tu {
            TimeUnit::Nanoseconds => {
                let vals = v
                    .into_iter()
                    .map(|opt_nd| opt_nd.and_then(|ndt| ndt.and_utc().timestamp_nanos_opt()));
                Int64Chunked::from_iter_options(name, vals).into_datetime(tu, None)
            },
            TimeUnit::Microseconds => {
                let vals = v
                    .into_iter()
                    .map(|opt_nd| opt_nd.map(datetime_to_timestamp_us));
                Int64Chunked::from_iter_options(name, vals).into_datetime(tu, None)
            },
            TimeUnit::Milliseconds => {
                let vals = v
                    .into_iter()
                    .map(|opt_nd| opt_nd.map(datetime_to_timestamp_ms));
                Int64Chunked::from_iter_options(name, vals).into_datetime(tu, None)
            },
        }
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
}

#[cfg(test)]
mod test {
    use chrono::{DateTime, NaiveDate, NaiveDateTime};

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

    #[test]
    fn from_datetime_nanosecond_boundaries() {
        let datetimes = [
            DateTime::from_timestamp_nanos(i64::MIN).naive_utc(),
            DateTime::from_timestamp_nanos(i64::MAX).naive_utc(),
        ];

        let dt = DatetimeChunked::from_naive_datetime(
            PlSmallStr::from_static("name"),
            datetimes,
            TimeUnit::Nanoseconds,
        );

        assert_eq!(dt.physical().cont_slice().unwrap(), &[i64::MIN, i64::MAX]);
    }

    #[test]
    fn from_datetime_out_of_nanosecond_range() {
        let valid = DateTime::from_timestamp_nanos(i64::MIN).naive_utc();
        let out_of_range = NaiveDate::from_ymd_opt(1600, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        assert_eq!(out_of_range.and_utc().timestamp_nanos_opt(), None);

        let dt = DatetimeChunked::from_naive_datetime(
            PlSmallStr::from_static("name"),
            [valid, out_of_range],
            TimeUnit::Nanoseconds,
        );
        assert_eq!(
            dt.physical().iter().collect::<Vec<_>>(),
            [Some(i64::MIN), None]
        );

        let dt = DatetimeChunked::from_naive_datetime_options(
            PlSmallStr::from_static("name"),
            [Some(valid), Some(out_of_range), None],
            TimeUnit::Nanoseconds,
        );
        assert_eq!(
            dt.physical().iter().collect::<Vec<_>>(),
            [Some(i64::MIN), None, None]
        );
    }

    #[test]
    fn from_datetime_non_nanosecond_units() {
        let datetime = DateTime::from_timestamp(1, 234_567_890)
            .unwrap()
            .naive_utc();

        for (time_unit, expected) in [
            (TimeUnit::Microseconds, 1_234_567),
            (TimeUnit::Milliseconds, 1_234),
        ] {
            let dt = DatetimeChunked::from_naive_datetime(
                PlSmallStr::from_static("name"),
                [datetime],
                time_unit,
            );
            assert_eq!(dt.physical().get(0), Some(expected));

            let dt = DatetimeChunked::from_naive_datetime_options(
                PlSmallStr::from_static("name"),
                [Some(datetime), None],
                time_unit,
            );
            assert_eq!(
                dt.physical().iter().collect::<Vec<_>>(),
                [Some(expected), None]
            );
        }
    }
}
