use std::fmt::Write;

use arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};
use chrono::format::{DelayedFormat, StrftimeItems};
use chrono::NaiveDate;
#[cfg(feature = "timezones")]
use chrono::TimeZone as TimeZoneTrait;
#[cfg(feature = "timezones")]
use chrono_tz::Tz;

use super::conversion::{datetime_to_timestamp_ms, datetime_to_timestamp_ns};
use super::*;
#[cfg(feature = "timezones")]
use crate::chunked_array::temporal::validate_time_zone;
use crate::prelude::DataType::Datetime;
use crate::prelude::*;

fn apply_datefmt_f<'a>(
    arr: &PrimitiveArray<i64>,
    fmted: &'a str,
    conversion_f: fn(i64) -> NaiveDateTime,
    datefmt_f: impl Fn(NaiveDateTime) -> DelayedFormat<StrftimeItems<'a>>,
) -> ArrayRef {
    let mut buf = String::new();
    let mut mutarr = MutableUtf8Array::with_capacities(arr.len(), arr.len() * fmted.len() + 1);
    for opt in arr.into_iter() {
        match opt {
            None => mutarr.push_null(),
            Some(v) => {
                buf.clear();
                let converted = conversion_f(*v);
                let datefmt = datefmt_f(converted);
                write!(buf, "{datefmt}").unwrap();
                mutarr.push(Some(&buf))
            },
        }
    }
    let arr: Utf8Array<i64> = mutarr.into();
    Box::new(arr)
}

#[cfg(feature = "timezones")]
fn format_tz(
    tz: Tz,
    arr: &PrimitiveArray<i64>,
    fmt: &str,
    fmted: &str,
    conversion_f: fn(i64) -> NaiveDateTime,
) -> ArrayRef {
    let datefmt_f = |ndt| tz.from_utc_datetime(&ndt).format(fmt);
    apply_datefmt_f(arr, fmted, conversion_f, datefmt_f)
}
fn format_naive(
    arr: &PrimitiveArray<i64>,
    fmt: &str,
    fmted: &str,
    conversion_f: fn(i64) -> NaiveDateTime,
) -> ArrayRef {
    let datefmt_f = |ndt: NaiveDateTime| ndt.format(fmt);
    apply_datefmt_f(arr, fmted, conversion_f, datefmt_f)
}

impl DatetimeChunked {
    pub fn as_datetime_iter(
        &self,
    ) -> impl Iterator<Item = Option<NaiveDateTime>> + TrustedLen + '_ {
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

    /// Convert from Datetime into Utf8 with the given format.
    /// See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    pub fn to_string(&self, format: &str) -> PolarsResult<Utf8Chunked> {
        #[cfg(feature = "timezones")]
        use chrono::Utc;
        let conversion_f = match self.time_unit() {
            TimeUnit::Nanoseconds => timestamp_ns_to_datetime,
            TimeUnit::Microseconds => timestamp_us_to_datetime,
            TimeUnit::Milliseconds => timestamp_ms_to_datetime,
        };

        let dt = NaiveDate::from_ymd_opt(2001, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let mut fmted = String::new();
        match self.time_zone() {
            #[cfg(feature = "timezones")]
            Some(_) => write!(
                fmted,
                "{}",
                Utc.from_local_datetime(&dt).earliest().unwrap().format(format)
            )
            .map_err(
                |_| polars_err!(ComputeError: "cannot format `DateTime` with format `{}`", format),
            )?,
            _ => write!(fmted, "{}", dt.format(format)).map_err(
                |_| polars_err!(ComputeError: "cannot format `NaiveDateTime` with format `{}`", format),
            )?,
        };
        let fmted = fmted; // discard mut

        let mut ca: Utf8Chunked = match self.time_zone() {
            #[cfg(feature = "timezones")]
            Some(time_zone) => self.apply_kernel_cast(&|arr| {
                format_tz(
                    time_zone.parse::<Tz>().unwrap(),
                    arr,
                    format,
                    &fmted,
                    conversion_f,
                )
            }),
            _ => self.apply_kernel_cast(&|arr| format_naive(arr, format, &fmted, conversion_f)),
        };
        ca.rename(self.name());
        Ok(ca)
    }

    /// Convert from Datetime into Utf8 with the given format.
    /// See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    ///
    /// Alias for `to_string`.
    pub fn strftime(&self, format: &str) -> PolarsResult<Utf8Chunked> {
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
                let ca = &self.0 / 1_000;
                out.0 = ca;
                out
            },
            (Nanoseconds, Milliseconds) => {
                let ca = &self.0 / 1_000_000;
                out.0 = ca;
                out
            },
            (Microseconds, Nanoseconds) => {
                let ca = &self.0 * 1_000;
                out.0 = ca;
                out
            },
            (Microseconds, Milliseconds) => {
                let ca = &self.0 / 1_000;
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
    pub fn set_time_unit(&mut self, tu: TimeUnit) {
        self.2 = Some(Datetime(tu, self.time_zone().clone()))
    }

    /// Change the underlying [`TimeZone`]. This does not modify the data.
    #[cfg(feature = "timezones")]
    pub fn set_time_zone(&mut self, time_zone: TimeZone) -> PolarsResult<()> {
        validate_time_zone(&time_zone)?;
        self.2 = Some(Datetime(self.time_unit(), Some(time_zone)));
        Ok(())
    }
    #[cfg(feature = "timezones")]
    pub fn convert_time_zone(mut self, time_zone: TimeZone) -> PolarsResult<Self> {
        polars_ensure!(
            self.time_zone().is_some(),
            InvalidOperation:
            "cannot call `convert_time_zone` on tz-naive; \
            set a time zone first with `replace_time_zone`"
        );
        self.set_time_zone(time_zone)?;
        Ok(self)
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
