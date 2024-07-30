pub mod infer;
use chrono::DateTime;
mod patterns;
mod strptime;
use chrono::format::ParseErrorKind;
use chrono::ParseError;
pub use patterns::Pattern;
#[cfg(feature = "dtype-time")]
use polars_core::chunked_array::temporal::time_to_time64ns;
use polars_core::prelude::arity::unary_elementwise;
use polars_utils::cache::FastCachedFunc;

use super::*;
#[cfg(feature = "dtype-date")]
use crate::chunkedarray::date::naive_date_to_date;
use crate::prelude::string::strptime::StrpTimeState;

#[cfg(feature = "dtype-time")]
fn time_pattern<F, K>(val: &str, convert: F) -> Option<&'static str>
// (string, fmt) -> PolarsResult
where
    F: Fn(&str, &str) -> chrono::ParseResult<K>,
{
    patterns::TIME_H_M_S
        .iter()
        .chain(patterns::TIME_H_M_S)
        .find(|fmt| convert(val, fmt).is_ok())
        .copied()
}

fn datetime_pattern<F, K>(val: &str, convert: F) -> Option<&'static str>
// (string, fmt) -> PolarsResult
where
    F: Fn(&str, &str) -> chrono::ParseResult<K>,
{
    patterns::DATETIME_Y_M_D
        .iter()
        .chain(patterns::DATETIME_D_M_Y)
        .find(|fmt| convert(val, fmt).is_ok())
        .copied()
}

fn date_pattern<F, K>(val: &str, convert: F) -> Option<&'static str>
// (string, fmt) -> PolarsResult
where
    F: Fn(&str, &str) -> chrono::ParseResult<K>,
{
    patterns::DATE_Y_M_D
        .iter()
        .chain(patterns::DATE_D_M_Y)
        .find(|fmt| convert(val, fmt).is_ok())
        .copied()
}

struct ParseErrorByteCopy(ParseErrorKind);

impl From<ParseError> for ParseErrorByteCopy {
    fn from(e: ParseError) -> Self {
        ParseErrorByteCopy(e.kind())
    }
}

fn get_first_val(ca: &StringChunked) -> PolarsResult<&str> {
    let idx = ca.first_non_null().ok_or_else(|| {
        polars_err!(ComputeError:
            "unable to determine date parsing format, all values are null",
        )
    })?;
    Ok(ca.get(idx).expect("should not be null"))
}

#[cfg(feature = "dtype-datetime")]
fn sniff_fmt_datetime(ca_string: &StringChunked) -> PolarsResult<&'static str> {
    let val = get_first_val(ca_string)?;
    datetime_pattern(val, NaiveDateTime::parse_from_str)
        .or_else(|| datetime_pattern(val, NaiveDate::parse_from_str))
        .ok_or_else(|| polars_err!(parse_fmt_idk = "datetime"))
}

#[cfg(feature = "dtype-date")]
fn sniff_fmt_date(ca_string: &StringChunked) -> PolarsResult<&'static str> {
    let val = get_first_val(ca_string)?;
    date_pattern(val, NaiveDate::parse_from_str).ok_or_else(|| polars_err!(parse_fmt_idk = "date"))
}

#[cfg(feature = "dtype-time")]
fn sniff_fmt_time(ca_string: &StringChunked) -> PolarsResult<&'static str> {
    let val = get_first_val(ca_string)?;
    time_pattern(val, NaiveTime::parse_from_str).ok_or_else(|| polars_err!(parse_fmt_idk = "time"))
}

pub trait StringMethods: AsString {
    #[cfg(feature = "dtype-time")]
    /// Parsing string values and return a [`TimeChunked`]
    fn as_time(&self, fmt: Option<&str>, use_cache: bool) -> PolarsResult<TimeChunked> {
        let string_ca = self.as_string();
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => sniff_fmt_time(string_ca)?,
        };
        let use_cache = use_cache && string_ca.len() > 50;

        let mut convert = FastCachedFunc::new(
            |s| {
                let naive_time = NaiveTime::parse_from_str(s, fmt).ok()?;
                Some(time_to_time64ns(&naive_time))
            },
            (string_ca.len() as f64).sqrt() as usize,
        );
        let ca = unary_elementwise(string_ca, |opt_s| convert.eval(opt_s?, use_cache));
        Ok(ca.with_name(string_ca.name()).into())
    }

    #[cfg(feature = "dtype-date")]
    /// Parsing string values and return a [`DateChunked`]
    /// Different from `as_date` this function allows matches that not contain the whole string
    /// e.g. "foo-2021-01-01-bar" could match "2021-01-01"
    fn as_date_not_exact(&self, fmt: Option<&str>) -> PolarsResult<DateChunked> {
        let string_ca = self.as_string();
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => sniff_fmt_date(string_ca)?,
        };
        let ca = unary_elementwise(string_ca, |opt_s| {
            let mut s = opt_s?;
            let fmt_len = fmt.len();

            for i in 1..(s.len().saturating_sub(fmt_len)) {
                if s.is_empty() {
                    return None;
                }
                match NaiveDate::parse_from_str(s, fmt).map(naive_date_to_date) {
                    Ok(nd) => return Some(nd),
                    Err(e) => match ParseErrorByteCopy::from(e).0 {
                        ParseErrorKind::TooLong => {
                            s = &s[..s.len() - 1];
                        },
                        _ => {
                            s = &s[i..];
                        },
                    },
                }
            }
            None
        });
        Ok(ca.with_name(string_ca.name()).into())
    }

    #[cfg(feature = "dtype-datetime")]
    /// Parsing string values and return a [`DatetimeChunked`]
    /// Different from `as_datetime` this function allows matches that not contain the whole string
    /// e.g. "foo-2021-01-01-bar" could match "2021-01-01"
    fn as_datetime_not_exact(
        &self,
        fmt: Option<&str>,
        tu: TimeUnit,
        tz_aware: bool,
        tz: Option<&TimeZone>,
        _ambiguous: &StringChunked,
    ) -> PolarsResult<DatetimeChunked> {
        let string_ca = self.as_string();
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => sniff_fmt_datetime(string_ca)?,
        };

        let func = match tu {
            TimeUnit::Nanoseconds => datetime_to_timestamp_ns,
            TimeUnit::Microseconds => datetime_to_timestamp_us,
            TimeUnit::Milliseconds => datetime_to_timestamp_ms,
        };

        let ca = unary_elementwise(string_ca, |opt_s| {
            let mut s = opt_s?;
            let fmt_len = fmt.len();

            for i in 1..(s.len().saturating_sub(fmt_len)) {
                if s.is_empty() {
                    return None;
                }
                let timestamp = if tz_aware {
                    DateTime::parse_from_str(s, fmt).map(|dt| func(dt.naive_utc()))
                } else {
                    NaiveDateTime::parse_from_str(s, fmt).map(func)
                };
                match timestamp {
                    Ok(ts) => return Some(ts),
                    Err(e) => {
                        let e: ParseErrorByteCopy = e.into();
                        match e.0 {
                            ParseErrorKind::TooLong => {
                                s = &s[..s.len() - 1];
                            },
                            _ => {
                                s = &s[i..];
                            },
                        }
                    },
                }
            }
            None
        })
        .with_name(string_ca.name());
        match (tz_aware, tz) {
            #[cfg(feature = "timezones")]
            (false, Some(tz)) => polars_ops::prelude::replace_time_zone(
                &ca.into_datetime(tu, None),
                Some(tz),
                _ambiguous,
                NonExistent::Raise,
            ),
            #[cfg(feature = "timezones")]
            (true, tz) => Ok(ca.into_datetime(tu, tz.cloned().or_else(|| Some("UTC".to_string())))),
            _ => Ok(ca.into_datetime(tu, None)),
        }
    }

    #[cfg(feature = "dtype-date")]
    /// Parsing string values and return a [`DateChunked`]
    fn as_date(&self, fmt: Option<&str>, use_cache: bool) -> PolarsResult<DateChunked> {
        let string_ca = self.as_string();
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => return infer::to_date(string_ca),
        };
        let use_cache = use_cache && string_ca.len() > 50;
        let fmt = strptime::compile_fmt(fmt)?;

        // We can use the fast parser.
        let ca = if let Some(fmt_len) = strptime::fmt_len(fmt.as_bytes()) {
            let mut strptime_cache = StrpTimeState::default();
            let mut convert = FastCachedFunc::new(
                |s: &str| {
                    // SAFETY: fmt_len is correct, it was computed with this `fmt` str.
                    match unsafe { strptime_cache.parse(s.as_bytes(), fmt.as_bytes(), fmt_len) } {
                        // Fallback to chrono.
                        None => NaiveDate::parse_from_str(s, &fmt).ok(),
                        Some(ndt) => Some(ndt.date()),
                    }
                    .map(naive_date_to_date)
                },
                (string_ca.len() as f64).sqrt() as usize,
            );
            unary_elementwise(string_ca, |val| convert.eval(val?, use_cache))
        } else {
            let mut convert = FastCachedFunc::new(
                |s| {
                    let naive_date = NaiveDate::parse_from_str(s, &fmt).ok()?;
                    Some(naive_date_to_date(naive_date))
                },
                (string_ca.len() as f64).sqrt() as usize,
            );
            unary_elementwise(string_ca, |val| convert.eval(val?, use_cache))
        };

        Ok(ca.with_name(string_ca.name()).into())
    }

    #[cfg(feature = "dtype-datetime")]
    /// Parsing string values and return a [`DatetimeChunked`].
    fn as_datetime(
        &self,
        fmt: Option<&str>,
        tu: TimeUnit,
        use_cache: bool,
        tz_aware: bool,
        tz: Option<&TimeZone>,
        ambiguous: &StringChunked,
    ) -> PolarsResult<DatetimeChunked> {
        let string_ca = self.as_string();
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => return infer::to_datetime(string_ca, tu, tz, ambiguous),
        };
        let fmt = strptime::compile_fmt(fmt)?;
        let use_cache = use_cache && string_ca.len() > 50;

        let func = match tu {
            TimeUnit::Nanoseconds => datetime_to_timestamp_ns,
            TimeUnit::Microseconds => datetime_to_timestamp_us,
            TimeUnit::Milliseconds => datetime_to_timestamp_ms,
        };

        if tz_aware {
            #[cfg(feature = "timezones")]
            {
                let mut convert = FastCachedFunc::new(
                    |s: &str| {
                        let dt = DateTime::parse_from_str(s, &fmt).ok()?;
                        Some(func(dt.naive_utc()))
                    },
                    (string_ca.len() as f64).sqrt() as usize,
                );
                Ok(
                    unary_elementwise(string_ca, |opt_s| convert.eval(opt_s?, use_cache))
                        .with_name(string_ca.name())
                        .into_datetime(
                            tu,
                            Some(tz.map(|x| x.to_string()).unwrap_or("UTC".to_string())),
                        ),
                )
            }
            #[cfg(not(feature = "timezones"))]
            {
                panic!("activate 'timezones' feature")
            }
        } else {
            let transform = match tu {
                TimeUnit::Nanoseconds => infer::transform_datetime_ns,
                TimeUnit::Microseconds => infer::transform_datetime_us,
                TimeUnit::Milliseconds => infer::transform_datetime_ms,
            };
            // We can use the fast parser.
            let ca = if let Some(fmt_len) = self::strptime::fmt_len(fmt.as_bytes()) {
                let mut strptime_cache = StrpTimeState::default();
                let mut convert = FastCachedFunc::new(
                    |s: &str| {
                        // SAFETY: fmt_len is correct, it was computed with this `fmt` str.
                        match unsafe { strptime_cache.parse(s.as_bytes(), fmt.as_bytes(), fmt_len) }
                        {
                            None => transform(s, &fmt),
                            Some(ndt) => Some(func(ndt)),
                        }
                    },
                    (string_ca.len() as f64).sqrt() as usize,
                );
                unary_elementwise(string_ca, |opt_s| convert.eval(opt_s?, use_cache))
            } else {
                let mut convert = FastCachedFunc::new(
                    |s| transform(s, &fmt),
                    (string_ca.len() as f64).sqrt() as usize,
                );
                unary_elementwise(string_ca, |opt_s| convert.eval(opt_s?, use_cache))
            };
            let dt = ca.with_name(string_ca.name()).into_datetime(tu, None);
            match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => polars_ops::prelude::replace_time_zone(
                    &dt,
                    Some(tz),
                    ambiguous,
                    NonExistent::Raise,
                ),
                _ => Ok(dt),
            }
        }
    }
}

pub trait AsString {
    fn as_string(&self) -> &StringChunked;
}

impl AsString for StringChunked {
    fn as_string(&self) -> &StringChunked {
        self
    }
}

impl StringMethods for StringChunked {}
