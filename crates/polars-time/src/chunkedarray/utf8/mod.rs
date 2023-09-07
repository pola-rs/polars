pub mod infer;
use chrono::DateTime;
mod patterns;
mod strptime;

use chrono::ParseError;
pub use patterns::Pattern;
#[cfg(feature = "dtype-time")]
use polars_core::chunked_array::temporal::time_to_time64ns;
use polars_utils::cache::CachedFunc;

use super::*;
#[cfg(feature = "dtype-date")]
use crate::chunkedarray::date::naive_date_to_date;
use crate::prelude::utf8::strptime::StrpTimeState;

#[cfg(feature = "dtype-time")]
fn time_pattern<F, K>(val: &str, convert: F) -> Option<&'static str>
// (string, fmt) -> PolarsResult
where
    F: Fn(&str, &str) -> chrono::ParseResult<K>,
{
    ["%T", "%T%.3f", "%T%.6f", "%T%.9f"]
        .into_iter()
        .find(|&fmt| convert(val, fmt).is_ok())
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
        // We need to do this until chrono ParseErrorKind is public
        // blocked by https://github.com/chronotope/chrono/pull/588.
        unsafe { std::mem::transmute(e) }
    }
}

#[allow(dead_code)]
enum ParseErrorKind {
    OutOfRange,
    Impossible,
    NotEnough,
    Invalid,
    /// The input string has been prematurely ended.
    TooShort,
    TooLong,
    BadFormat,
}

fn get_first_val(ca: &Utf8Chunked) -> PolarsResult<&str> {
    let idx = ca.first_non_null().ok_or_else(|| {
        polars_err!(ComputeError:
            "unable to determine date parsing format, all values are null",
        )
    })?;
    Ok(ca.get(idx).expect("should not be null"))
}

#[cfg(feature = "dtype-datetime")]
fn sniff_fmt_datetime(ca_utf8: &Utf8Chunked) -> PolarsResult<&'static str> {
    let val = get_first_val(ca_utf8)?;
    datetime_pattern(val, NaiveDateTime::parse_from_str)
        .or_else(|| datetime_pattern(val, NaiveDate::parse_from_str))
        .ok_or_else(|| polars_err!(parse_fmt_idk = "datetime"))
}

#[cfg(feature = "dtype-date")]
fn sniff_fmt_date(ca_utf8: &Utf8Chunked) -> PolarsResult<&'static str> {
    let val = get_first_val(ca_utf8)?;
    date_pattern(val, NaiveDate::parse_from_str).ok_or_else(|| polars_err!(parse_fmt_idk = "date"))
}

#[cfg(feature = "dtype-time")]
fn sniff_fmt_time(ca_utf8: &Utf8Chunked) -> PolarsResult<&'static str> {
    let val = get_first_val(ca_utf8)?;
    time_pattern(val, NaiveTime::parse_from_str).ok_or_else(|| polars_err!(parse_fmt_idk = "time"))
}

pub trait Utf8Methods: AsUtf8 {
    #[cfg(feature = "dtype-time")]
    /// Parsing string values and return a [`TimeChunked`]
    fn as_time(&self, fmt: Option<&str>, use_cache: bool) -> PolarsResult<TimeChunked> {
        let utf8_ca = self.as_utf8();
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => sniff_fmt_time(utf8_ca)?,
        };
        let use_cache = use_cache && utf8_ca.len() > 50;

        let mut convert = CachedFunc::new(|s| {
            let naive_time = NaiveTime::parse_from_str(s, fmt).ok()?;
            Some(time_to_time64ns(&naive_time))
        });
        let ca = utf8_ca.apply_generic(|opt_s| convert.eval(opt_s?, use_cache));
        Ok(ca.with_name(utf8_ca.name()).into())
    }

    #[cfg(feature = "dtype-date")]
    /// Parsing string values and return a [`DateChunked`]
    /// Different from `as_date` this function allows matches that not contain the whole string
    /// e.g. "foo-2021-01-01-bar" could match "2021-01-01"
    fn as_date_not_exact(&self, fmt: Option<&str>) -> PolarsResult<DateChunked> {
        let utf8_ca = self.as_utf8();
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => sniff_fmt_date(utf8_ca)?,
        };
        let ca = utf8_ca.apply_generic(|opt_s| {
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
        Ok(ca.with_name(utf8_ca.name()).into())
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
        _ambiguous: &Utf8Chunked,
    ) -> PolarsResult<DatetimeChunked> {
        let utf8_ca = self.as_utf8();
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => sniff_fmt_datetime(utf8_ca)?,
        };

        let func = match tu {
            TimeUnit::Nanoseconds => datetime_to_timestamp_ns,
            TimeUnit::Microseconds => datetime_to_timestamp_us,
            TimeUnit::Milliseconds => datetime_to_timestamp_ms,
        };

        let ca = utf8_ca
            .apply_generic(|opt_s| {
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
            .with_name(utf8_ca.name());
        match (tz_aware, tz) {
            #[cfg(feature = "timezones")]
            (false, Some(tz)) => polars_ops::prelude::replace_time_zone(
                &ca.into_datetime(tu, None),
                Some(tz),
                _ambiguous,
            ),
            #[cfg(feature = "timezones")]
            (true, _) => Ok(ca.into_datetime(tu, Some("UTC".to_string()))),
            _ => Ok(ca.into_datetime(tu, None)),
        }
    }

    #[cfg(feature = "dtype-date")]
    /// Parsing string values and return a [`DateChunked`]
    fn as_date(&self, fmt: Option<&str>, use_cache: bool) -> PolarsResult<DateChunked> {
        let utf8_ca = self.as_utf8();
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => return infer::to_date(utf8_ca),
        };
        let use_cache = use_cache && utf8_ca.len() > 50;
        let fmt = strptime::compile_fmt(fmt)?;

        // We can use the fast parser.
        let ca = if let Some(fmt_len) = strptime::fmt_len(fmt.as_bytes()) {
            let mut strptime_cache = StrpTimeState::default();
            let mut convert = CachedFunc::new(|s: &str| {
                // SAFETY: fmt_len is correct, it was computed with this `fmt` str.
                match unsafe { strptime_cache.parse(s.as_bytes(), fmt.as_bytes(), fmt_len) } {
                    // Fallback to chrono.
                    None => NaiveDate::parse_from_str(s, &fmt).ok(),
                    Some(ndt) => Some(ndt.date()),
                }
                .map(naive_date_to_date)
            });
            utf8_ca.apply_generic(|val| convert.eval(val?, use_cache))
        } else {
            let mut convert = CachedFunc::new(|s| {
                let naive_date = NaiveDate::parse_from_str(s, &fmt).ok()?;
                Some(naive_date_to_date(naive_date))
            });
            utf8_ca.apply_generic(|val| convert.eval(val?, use_cache))
        };

        Ok(ca.with_name(utf8_ca.name()).into())
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
        ambiguous: &Utf8Chunked,
    ) -> PolarsResult<DatetimeChunked> {
        let utf8_ca = self.as_utf8();
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => return infer::to_datetime(utf8_ca, tu, tz, ambiguous),
        };
        let fmt = strptime::compile_fmt(fmt)?;
        let use_cache = use_cache && utf8_ca.len() > 50;

        let func = match tu {
            TimeUnit::Nanoseconds => datetime_to_timestamp_ns,
            TimeUnit::Microseconds => datetime_to_timestamp_us,
            TimeUnit::Milliseconds => datetime_to_timestamp_ms,
        };

        if tz_aware {
            #[cfg(feature = "timezones")]
            {
                let mut convert = CachedFunc::new(|s: &str| {
                    let dt = DateTime::parse_from_str(s, &fmt).ok()?;
                    Some(func(dt.naive_utc()))
                });
                Ok(utf8_ca
                    .apply_generic(|opt_s| convert.eval(opt_s?, use_cache))
                    .with_name(utf8_ca.name())
                    .into_datetime(tu, Some("UTC".to_string())))
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
                let mut convert = CachedFunc::new(|s: &str| {
                    // SAFETY: fmt_len is correct, it was computed with this `fmt` str.
                    match unsafe { strptime_cache.parse(s.as_bytes(), fmt.as_bytes(), fmt_len) } {
                        None => transform(s, &fmt),
                        Some(ndt) => Some(func(ndt)),
                    }
                });
                utf8_ca.apply_generic(|opt_s| convert.eval(opt_s?, use_cache))
            } else {
                let mut convert = CachedFunc::new(|s| transform(s, &fmt));
                utf8_ca.apply_generic(|opt_s| convert.eval(opt_s?, use_cache))
            };
            let dt = ca.with_name(utf8_ca.name()).into_datetime(tu, None);
            match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => polars_ops::prelude::replace_time_zone(&dt, Some(tz), ambiguous),
                _ => Ok(dt),
            }
        }
    }
}

pub trait AsUtf8 {
    fn as_utf8(&self) -> &Utf8Chunked;
}

impl AsUtf8 for Utf8Chunked {
    fn as_utf8(&self) -> &Utf8Chunked {
        self
    }
}

impl Utf8Methods for Utf8Chunked {}
