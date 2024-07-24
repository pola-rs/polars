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
            self.downcast_iter()
                .flat_map(move |iter| iter.into_iter().map(move |opt_v| opt_v.copied().map(func)))
                .trust_my_length(self.len())
        }
    }

    pub fn time_unit(&self) -> TimeUnit {
        match self.2.as_ref().unwrap() {
            DataType::Datetime(tu, _) => *tu,
            _ => unreachable!(),
        }
    }

    pub fn time_zone(&self) -> &Option<TimeZone> {
        match self.2.as_ref().unwrap() {
            DataType::Datetime(_, tz) => tz,
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

        let mut ca: StringChunked = match self.time_zone() {
            #[cfg(feature = "timezones")]
            Some(time_zone) => {
                let parsed_time_zone = time_zone.parse::<Tz>().expect("already validated");
                let datefmt_f = |ndt| parsed_time_zone.from_utc_datetime(&ndt).format(format);
                self.try_apply_into_string_amortized(|val, buf| {
                    let ndt = conversion_f(val);
                    write!(buf, "{}", datefmt_f(ndt))
                    }
                ).map_err(
                |_| polars_err!(ComputeError: "cannot format timezone-aware Datetime with format '{}'", format),
                )?
            },
            _ => {
                let datefmt_f = |ndt: NaiveDateTime| ndt.format(format);
                self.try_apply_into_string_amortized(|val, buf| {
                    let ndt = conversion_f(val);
                    write!(buf, "{}", datefmt_f(ndt))
                    }
                ).map_err(
                |_| polars_err!(ComputeError: "cannot format timezone-naive Datetime with format '{}'", format),
                )?
            },
        };
        ca.rename(self.name());
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
        name: &str,
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
        name: &str,
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

        use TimeUnit::*;
        match (current_unit, tu) {
            (Nanoseconds, Microseconds) => {
                let ca = (&self.0).wrapping_trunc_div_scalar(1_000);
                out.0 = ca;
                out
            },
            (Nanoseconds, Milliseconds) => {
                let ca = (&self.0).wrapping_trunc_div_scalar(1_000_000);
                out.0 = ca;
                out
            },
            (Microseconds, Nanoseconds) => {
                let ca = &self.0 * 1_000;
                out.0 = ca;
                out
            },
            (Microseconds, Milliseconds) => {
                let ca = (&self.0).wrapping_trunc_div_scalar(1_000);
                out.0 = ca;
                out
            },
            (Milliseconds, Nanoseconds) => {
                let ca = &self.0 * 1_000_000;
                out.0 = ca;
                out
            },
            (Milliseconds, Microseconds) => {
                let ca = &self.0 * 1_000;
                out.0 = ca;
                out
            },
            (Nanoseconds, Nanoseconds)
            | (Microseconds, Microseconds)
            | (Milliseconds, Milliseconds) => out,
        }
    }

    /// Change the underlying [`TimeUnit`]. This does not modify the data.
    pub fn set_time_unit(&mut self, time_unit: TimeUnit) {
        self.2 = Some(Datetime(time_unit, self.time_zone().clone()))
    }

    /// Change the underlying [`TimeZone`]. This does not modify the data.
    /// This does not validate the time zone - it's up to the caller to verify that it's
    /// already been validated.
    #[cfg(feature = "timezones")]
    pub fn set_time_zone(&mut self, time_zone: TimeZone) -> PolarsResult<()> {
        self.2 = Some(Datetime(self.time_unit(), Some(time_zone)));
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
        self.2 = Some(Datetime(time_unit, Some(time_zone)));
        Ok(())
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
            "name",
            datetimes.iter().copied(),
            TimeUnit::Nanoseconds,
        );
        assert_eq!(
            [
                588_470_416_000_000_000,
                1_441_497_364_000_000_000,
                1_356_048_000_000_000_000
            ],
            dt.cont_slice().unwrap()
        );
    }
}
