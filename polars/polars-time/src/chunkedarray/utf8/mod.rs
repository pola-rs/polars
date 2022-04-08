pub mod infer;
mod patterns;
mod strptime;

use super::*;
#[cfg(feature = "dtype-date")]
use crate::chunkedarray::date::naive_date_to_date;
#[cfg(feature = "dtype-time")]
use crate::chunkedarray::time::time_to_time64ns;
use chrono::ParseError;
pub use patterns::Pattern;

#[cfg(feature = "dtype-time")]
fn time_pattern<F, K>(val: &str, convert: F) -> Option<&'static str>
// (string, fmt) -> result
where
    F: Fn(&str, &str) -> chrono::ParseResult<K>,
{
    for fmt in ["%T", "%T%.3f", "%T%.6f", "%T%.9f"] {
        if convert(val, fmt).is_ok() {
            return Some(fmt);
        }
    }
    None
}

fn datetime_pattern<F, K>(val: &str, convert: F) -> Option<&'static str>
// (string, fmt) -> result
where
    F: Fn(&str, &str) -> chrono::ParseResult<K>,
{
    for fmt in [
        // 21/12/31 12:54:98
        "%y/%m/%d %H:%M:%S",
        // 2021-12-31 24:58:01
        "%Y-%m-%d %H:%M:%S",
        // 21/12/31 24:58:01
        "%y/%m/%d %H:%M:%S",
        //210319 23:58:50
        "%y%m%d %H:%M:%S",
        // 2019-04-18T02:45:55
        // 2021/12/31 12:54:98
        "%Y/%m/%d %H:%M:%S",
        // 2021-12-31 24:58:01
        "%Y-%m-%d %H:%M:%S",
        // 2021/12/31 24:58:01
        "%Y/%m/%d %H:%M:%S",
        // 20210319 23:58:50
        "%Y%m%d %H:%M:%S",
        // 2019-04-18T02:45:55
        // %F cannot be parse by polars native parser
        "%Y-%m-%dT%H:%M:%S",
        // 2019-04-18T02:45:55.555000000
        // microseconds
        "%Y-%m-%dT%H:%M:%S.%6f",
        // nanoseconds
        "%Y-%m-%dT%H:%M:%S.%9f",
    ] {
        if convert(val, fmt).is_ok() {
            return Some(fmt);
        }
    }
    None
}

fn date_pattern<F, K>(val: &str, convert: F) -> Option<&'static str>
// (string, fmt) -> result
where
    F: Fn(&str, &str) -> chrono::ParseResult<K>,
{
    for fmt in [
        // 2021-12-31
        "%Y-%m-%d", // 31-12-2021
        "%d-%m-%Y", // 2021319 (2021-03-19)
        "%Y%m%d",
    ] {
        if convert(val, fmt).is_ok() {
            return Some(fmt);
        }
    }
    None
}

struct ParseErrorByteCopy(ParseErrorKind);

impl From<ParseError> for ParseErrorByteCopy {
    fn from(e: ParseError) -> Self {
        // we need to do this until chrono ParseErrorKind is public
        // blocked by https://github.com/chronotope/chrono/pull/588
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

pub trait Utf8Methods {
    #[cfg(feature = "dtype-time")]
    /// Parsing string values and return a [`TimeChunked`]
    fn as_time(&self, fmt: Option<&str>) -> Result<TimeChunked>;

    #[cfg(feature = "dtype-date")]
    /// Parsing string values and return a [`DateChunked`]
    /// Different from `as_date` this function allows matches that not contain the whole string
    /// e.g. "foo-2021-01-01-bar" could match "2021-01-01"
    fn as_date_not_exact(&self, fmt: Option<&str>) -> Result<DateChunked>;

    #[cfg(feature = "dtype-datetime")]
    /// Parsing string values and return a [`DatetimeChunked`]
    /// Different from `as_datetime` this function allows matches that not contain the whole string
    /// e.g. "foo-2021-01-01-bar" could match "2021-01-01"
    fn as_datetime_not_exact(&self, fmt: Option<&str>, tu: TimeUnit) -> Result<DatetimeChunked>;

    #[cfg(feature = "dtype-date")]
    /// Parsing string values and return a [`DateChunked`]
    fn as_date(&self, fmt: Option<&str>) -> Result<DateChunked>;

    #[cfg(feature = "dtype-datetime")]
    /// Parsing string values and return a [`DatetimeChunked`]
    fn as_datetime(&self, fmt: Option<&str>, tu: TimeUnit) -> Result<DatetimeChunked>;
}

fn get_first_val(ca: &Utf8Chunked) -> Result<&str> {
    let idx = match ca.first_non_null() {
        Some(idx) => idx,
        None => {
            return Err(PolarsError::ComputeError(
                "Cannot determine date parsing format, all values are null".into(),
            ))
        }
    };
    let val = ca.get(idx).expect("should not be null");
    Ok(val)
}

#[cfg(feature = "dtype-datetime")]
fn sniff_fmt_datetime(ca_utf8: &Utf8Chunked) -> Result<&'static str> {
    let val = get_first_val(ca_utf8)?;
    if let Some(pattern) = datetime_pattern(val, NaiveDateTime::parse_from_str) {
        return Ok(pattern);
    }
    Err(PolarsError::ComputeError(
        "Could not find an appropriate format to parse dates, please define a fmt".into(),
    ))
}

#[cfg(feature = "dtype-date")]
fn sniff_fmt_date(ca_utf8: &Utf8Chunked) -> Result<&'static str> {
    let val = get_first_val(ca_utf8)?;
    if let Some(pattern) = date_pattern(val, NaiveDate::parse_from_str) {
        return Ok(pattern);
    }
    Err(PolarsError::ComputeError(
        "Could not find an appropriate format to parse dates, please define a fmt".into(),
    ))
}

#[cfg(feature = "dtype-time")]
fn sniff_fmt_time(ca_utf8: &Utf8Chunked) -> Result<&'static str> {
    let val = get_first_val(ca_utf8)?;
    if let Some(pattern) = time_pattern(val, NaiveTime::parse_from_str) {
        return Ok(pattern);
    }
    Err(PolarsError::ComputeError(
        "Could not find an appropriate format to parse times, please define a fmt".into(),
    ))
}

impl Utf8Methods for Utf8Chunked {
    #[cfg(feature = "dtype-time")]
    /// Parsing string values and return a [`TimeChunked`]
    fn as_time(&self, fmt: Option<&str>) -> Result<TimeChunked> {
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => sniff_fmt_time(self)?,
        };

        let mut ca: Int64Chunked = match self.has_validity() {
            false => self
                .into_no_null_iter()
                .map(|s| {
                    NaiveTime::parse_from_str(s, fmt)
                        .ok()
                        .as_ref()
                        .map(time_to_time64ns)
                })
                .collect_trusted(),
            _ => self
                .into_iter()
                .map(|opt_s| {
                    let opt_nd = opt_s.map(|s| {
                        NaiveTime::parse_from_str(s, fmt)
                            .ok()
                            .as_ref()
                            .map(time_to_time64ns)
                    });
                    match opt_nd {
                        None => None,
                        Some(None) => None,
                        Some(Some(nd)) => Some(nd),
                    }
                })
                .collect_trusted(),
        };
        ca.rename(self.name());
        Ok(ca.into())
    }

    #[cfg(feature = "dtype-date")]
    /// Parsing string values and return a [`DateChunked`]
    /// Different from `as_date` this function allows matches that not contain the whole string
    /// e.g. "foo-2021-01-01-bar" could match "2021-01-01"
    fn as_date_not_exact(&self, fmt: Option<&str>) -> Result<DateChunked> {
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => sniff_fmt_date(self)?,
        };
        let mut ca: Int32Chunked = self
            .into_iter()
            .map(|opt_s| match opt_s {
                None => None,
                Some(mut s) => {
                    let fmt_len = fmt.len();

                    for i in 1..(s.len() - fmt_len) {
                        if s.is_empty() {
                            return None;
                        }
                        match NaiveDate::parse_from_str(s, fmt).map(naive_date_to_date) {
                            Ok(nd) => return Some(nd),
                            Err(e) => {
                                let e: ParseErrorByteCopy = e.into();
                                match e.0 {
                                    ParseErrorKind::TooLong => {
                                        s = &s[..s.len() - 1];
                                    }
                                    _ => {
                                        s = &s[i..];
                                    }
                                }
                            }
                        }
                    }
                    None
                }
            })
            .collect_trusted();
        ca.rename(self.name());
        Ok(ca.into())
    }

    #[cfg(feature = "dtype-datetime")]
    /// Parsing string values and return a [`DatetimeChunked`]
    /// Different from `as_datetime` this function allows matches that not contain the whole string
    /// e.g. "foo-2021-01-01-bar" could match "2021-01-01"
    fn as_datetime_not_exact(&self, fmt: Option<&str>, tu: TimeUnit) -> Result<DatetimeChunked> {
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => sniff_fmt_datetime(self)?,
        };

        let func = match tu {
            TimeUnit::Nanoseconds => datetime_to_timestamp_ns,
            TimeUnit::Microseconds => datetime_to_timestamp_us,
            TimeUnit::Milliseconds => datetime_to_timestamp_ms,
        };

        let mut ca: Int64Chunked = self
            .into_iter()
            .map(|opt_s| match opt_s {
                None => None,
                Some(mut s) => {
                    let fmt_len = fmt.len();

                    for i in 1..(s.len() - fmt_len) {
                        if s.is_empty() {
                            return None;
                        }
                        match NaiveDateTime::parse_from_str(s, fmt).map(func) {
                            Ok(nd) => return Some(nd),
                            Err(e) => {
                                let e: ParseErrorByteCopy = e.into();
                                match e.0 {
                                    ParseErrorKind::TooLong => {
                                        s = &s[..s.len() - 1];
                                    }
                                    _ => {
                                        s = &s[i..];
                                    }
                                }
                            }
                        }
                    }
                    None
                }
            })
            .collect_trusted();
        ca.rename(self.name());
        Ok(ca.into_datetime(tu, None))
    }

    #[cfg(feature = "dtype-date")]
    /// Parsing string values and return a [`DateChunked`]
    fn as_date(&self, fmt: Option<&str>) -> Result<DateChunked> {
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => return infer::to_date(self),
        };
        let fmt = self::strptime::compile_fmt(fmt);

        // we can use the fast parser
        let mut ca: Int32Chunked = if let Some(fmt_len) = self::strptime::fmt_len(fmt.as_bytes()) {
            let convert = |s: &str| {
                // Safety:
                // fmt_len is correct, it was computed with this `fmt` str.
                match unsafe { self::strptime::parse(s.as_bytes(), fmt.as_bytes(), fmt_len) } {
                    // fallback to chrono
                    None => NaiveDate::parse_from_str(s, &fmt).ok(),
                    Some(ndt) => Some(ndt.date()),
                }
                .map(naive_date_to_date)
            };

            if self.null_count() == 0 {
                self.into_no_null_iter().map(convert).collect_trusted()
            } else {
                self.into_iter()
                    .map(|opt_s| opt_s.and_then(convert))
                    .collect_trusted()
            }
        } else {
            self.into_iter()
                .map(|opt_s| {
                    opt_s.and_then(|s| {
                        NaiveDate::parse_from_str(s, &fmt)
                            .ok()
                            .map(naive_date_to_date)
                    })
                })
                .collect_trusted()
        };

        ca.rename(self.name());
        Ok(ca.into())
    }

    #[cfg(feature = "dtype-datetime")]
    /// Parsing string values and return a [`DatetimeChunked`]
    fn as_datetime(&self, fmt: Option<&str>, tu: TimeUnit) -> Result<DatetimeChunked> {
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => return infer::to_datetime(self, tu),
        };
        let fmt = self::strptime::compile_fmt(fmt);

        let func = match tu {
            TimeUnit::Nanoseconds => datetime_to_timestamp_ns,
            TimeUnit::Microseconds => datetime_to_timestamp_us,
            TimeUnit::Milliseconds => datetime_to_timestamp_ms,
        };

        // we can use the fast parser
        let mut ca: Int64Chunked = if let Some(fmt_len) = self::strptime::fmt_len(fmt.as_bytes()) {
            let convert = |s: &str| {
                // Safety:
                // fmt_len is correct, it was computed with this `fmt` str.
                match unsafe { self::strptime::parse(s.as_bytes(), fmt.as_bytes(), fmt_len) } {
                    // fallback to chrono
                    None => NaiveDateTime::parse_from_str(s, &fmt).ok(),
                    Some(v) => Some(v),
                }
                .map(func)
            };
            if self.null_count() == 0 {
                self.into_no_null_iter().map(convert).collect_trusted()
            } else {
                self.into_iter()
                    .map(|opt_s| opt_s.and_then(convert))
                    .collect_trusted()
            }
        } else {
            self.into_iter()
                .map(|opt_s| {
                    opt_s.and_then(|s| NaiveDateTime::parse_from_str(s, &fmt).ok().map(func))
                })
                .collect_trusted()
        };

        ca.rename(self.name());
        Ok(ca.into_datetime(tu, None))
    }
}
