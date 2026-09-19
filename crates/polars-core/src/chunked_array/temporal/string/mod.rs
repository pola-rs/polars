pub mod infer;
pub mod patterns;
pub mod strptime;

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime};
use polars_utils::cache::LruCachedFunc;

use self::strptime::StrpTimeState;
use crate::chunked_array::ops::arity::unary_elementwise;
#[cfg(feature = "dtype-date")]
use crate::chunked_array::temporal::date::naive_date_to_date;
#[cfg(feature = "dtype-time")]
use crate::chunked_array::temporal::time_to_time64ns;
use crate::prelude::*;

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

/// Localize before converting to the target unit: local wall time may lie outside
/// the physical timestamp range even when the corresponding UTC instant fits.
#[cfg(all(feature = "dtype-datetime", feature = "timezones"))]
fn parse_local_datetime<'a>(
    ca: &'a StringChunked,
    tu: TimeUnit,
    tz: &TimeZone,
    ambiguous: &'a StringChunked,
    mut parse: impl FnMut(&'a str) -> Option<NaiveDateTime>,
) -> PolarsResult<DatetimeChunked> {
    use either::Either;
    use polars_arrow::legacy::kernels::convert_to_naive_local;

    let time_zone = tz.to_chrono()?;
    polars_ensure!(
        ca.len() == ambiguous.len() || ca.len() == 1 || ambiguous.len() == 1,
        length_mismatch = "strptime",
        ca.len(),
        ambiguous.len()
    );
    let len = if ca.len() == 1 {
        ambiguous.len()
    } else {
        ca.len()
    };
    let values = if ca.len() == 1 {
        Either::Left(std::iter::repeat_n(ca.get(0), len))
    } else {
        Either::Right(ca.iter())
    };
    let ambiguities = if ambiguous.len() == 1 {
        Either::Left(std::iter::repeat_n(ambiguous.get(0), len))
    } else {
        Either::Right(ambiguous.iter())
    };
    let out: Int64Chunked = values
        .zip(ambiguities)
        .map(|(value, ambiguous)| {
            let (Some(value), Some(ambiguous)) = (value, ambiguous) else {
                return Ok::<_, PolarsError>(None);
            };
            let Some(dt) = parse(value) else {
                return Ok(None);
            };
            let Some(dt) = convert_to_naive_local(
                &chrono_tz::UTC,
                &time_zone,
                dt,
                ambiguous.parse()?,
                NonExistent::Raise,
            )?
            else {
                return Ok(None);
            };
            Ok(match tu {
                TimeUnit::Nanoseconds => dt.and_utc().timestamp_nanos_opt(),
                TimeUnit::Microseconds => Some(datetime_to_timestamp_us(dt)),
                TimeUnit::Milliseconds => Some(datetime_to_timestamp_ms(dt)),
            })
        })
        .collect::<PolarsResult<_>>()?;
    Ok(out
        .with_name(ca.name().clone())
        .into_datetime(tu, Some(tz.clone())))
}

pub trait StringMethods: AsString {
    #[cfg(feature = "dtype-time")]
    /// Parsing string values and return a [`TimeChunked`]
    fn as_time(&self, fmt: Option<&str>, use_cache: bool) -> PolarsResult<TimeChunked> {
        let string_ca = self.as_string();
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => {
                if string_ca.null_count() == string_ca.len() {
                    return Ok(
                        Int64Chunked::full_null(string_ca.name().clone(), string_ca.len())
                            .into_time(),
                    );
                }
                infer::infer_from_values(string_ca, |val| {
                    time_pattern(val, NaiveTime::parse_from_str)
                })
                .ok_or_else(|| polars_err!(parse_fmt_idk = "time"))?
            },
        };
        let use_cache = use_cache && string_ca.len() > 50;

        let mut convert = LruCachedFunc::new(
            |s| {
                let naive_time = NaiveTime::parse_from_str(s, fmt).ok()?;
                Some(time_to_time64ns(&naive_time))
            },
            (string_ca.len() as f64).sqrt() as usize,
        );
        let ca = unary_elementwise(string_ca, |opt_s| convert.eval(opt_s?, use_cache));
        Ok(ca.with_name(string_ca.name().clone()).into_time())
    }

    #[cfg(feature = "dtype-date")]
    /// Parsing string values and return a [`DateChunked`]
    /// Different from `as_date` this function allows matches that not contain the whole string
    /// e.g. "foo-2021-01-01-bar" could match "2021-01-01"
    fn as_date_not_exact(&self, fmt: Option<&str>) -> PolarsResult<DateChunked> {
        let string_ca = self.as_string();
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => {
                if string_ca.null_count() == string_ca.len() {
                    return Ok(
                        Int32Chunked::full_null(string_ca.name().clone(), string_ca.len())
                            .into_date(),
                    );
                }
                infer::infer_from_values(string_ca, |val| {
                    date_pattern(val, NaiveDate::parse_from_str)
                })
                .ok_or_else(|| polars_err!(parse_fmt_idk = "date"))?
            },
        };
        let ca = unary_elementwise(string_ca, |opt_s| {
            let mut s = opt_s?;
            while !s.is_empty() {
                match NaiveDate::parse_and_remainder(s, fmt) {
                    Ok((nd, _)) => return Some(naive_date_to_date(nd)),
                    Err(_) => {
                        let mut it = s.chars();
                        it.next();
                        s = it.as_str();
                    },
                }
            }

            None
        });
        Ok(ca.with_name(string_ca.name().clone()).into_date())
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
        // Ensure that the inferred time_zone matches the given time_zone.
        ensure_matching_tz: bool,
    ) -> PolarsResult<DatetimeChunked> {
        let string_ca = self.as_string();
        let had_format = fmt.is_some();
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => {
                if string_ca.null_count() == string_ca.len() {
                    return Ok(
                        Int64Chunked::full_null(string_ca.name().clone(), string_ca.len())
                            .into_datetime(tu, tz.cloned()),
                    );
                }
                infer::infer_from_values(string_ca, |val| {
                    datetime_pattern(val, NaiveDateTime::parse_from_str)
                        .or_else(|| datetime_pattern(val, NaiveDate::parse_from_str))
                })
                .ok_or_else(|| polars_err!(parse_fmt_idk = "datetime"))?
            },
        };

        #[cfg(feature = "timezones")]
        if !tz_aware
            && let Some(tz) = tz
            && tz != &TimeZone::UTC
        {
            return parse_local_datetime(string_ca, tu, tz, _ambiguous, |mut s| {
                while !s.is_empty() {
                    if let Some((dt, _)) = infer::parse_datetime_and_remainder(s, fmt) {
                        return Some(dt);
                    }
                    let mut chars = s.chars();
                    chars.next();
                    s = chars.as_str();
                }
                None
            });
        }

        let func: fn(NaiveDateTime) -> Option<i64> = match tu {
            TimeUnit::Nanoseconds => |dt| dt.and_utc().timestamp_nanos_opt(),
            TimeUnit::Microseconds => |dt| Some(datetime_to_timestamp_us(dt)),
            TimeUnit::Milliseconds => |dt| Some(datetime_to_timestamp_ms(dt)),
        };

        let ca = unary_elementwise(string_ca, |opt_s| {
            let mut s = opt_s?;
            while !s.is_empty() {
                let datetime = if tz_aware {
                    DateTime::parse_and_remainder(s, fmt)
                        .ok()
                        .map(|(dt, _r)| dt.naive_utc())
                } else {
                    infer::parse_datetime_and_remainder(s, fmt).map(|(nd, _r)| nd)
                };
                match datetime {
                    Some(dt) => return func(dt),
                    None => {
                        let mut it = s.chars();
                        it.next();
                        s = it.as_str();
                    },
                }
            }
            None
        })
        .with_name(string_ca.name().clone());

        polars_ensure!(
            !ensure_matching_tz || had_format || !(tz_aware && tz.is_none()),
            to_datetime_tz_mismatch
        );

        match (tz_aware, tz) {
            #[cfg(feature = "timezones")]
            (false, Some(tz)) => {
                crate::chunked_array::temporal::replace_time_zone::replace_time_zone(
                    &ca.into_datetime(tu, None),
                    Some(tz),
                    _ambiguous,
                    NonExistent::Raise,
                )
            },
            #[cfg(feature = "timezones")]
            (true, tz) => Ok(ca.into_datetime(tu, Some(tz.cloned().unwrap_or(TimeZone::UTC)))),
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
        let ca = if strptime::fast_parser_supported(fmt.as_bytes()) {
            let mut strptime_cache = StrpTimeState::default();
            let mut convert = LruCachedFunc::new(
                |s: &str| {
                    match strptime_cache.parse(s.as_bytes(), fmt.as_bytes()) {
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
            let mut convert = LruCachedFunc::new(
                |s| {
                    let naive_date = NaiveDate::parse_from_str(s, &fmt).ok()?;
                    Some(naive_date_to_date(naive_date))
                },
                (string_ca.len() as f64).sqrt() as usize,
            );
            unary_elementwise(string_ca, |val| convert.eval(val?, use_cache))
        };

        Ok(ca.with_name(string_ca.name().clone()).into_date())
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
            None => return infer::to_datetime(string_ca, tu, tz, ambiguous, true),
        };
        let fmt = strptime::compile_fmt(fmt)?;
        let use_cache = use_cache && string_ca.len() > 50;

        #[cfg(feature = "timezones")]
        if !tz_aware
            && let Some(tz) = tz
            && tz != &TimeZone::UTC
        {
            let mut strptime_cache = StrpTimeState::default();
            let mut convert = LruCachedFunc::new(
                |s: &str| {
                    strptime_cache
                        .parse(s.as_bytes(), fmt.as_bytes())
                        .or_else(|| infer::parse_datetime(s, &fmt))
                },
                (string_ca.len() as f64).sqrt() as usize,
            );
            return parse_local_datetime(string_ca, tu, tz, ambiguous, |s| {
                convert.eval(s, use_cache)
            });
        }

        let func: fn(NaiveDateTime) -> Option<i64> = match tu {
            TimeUnit::Nanoseconds => |dt| dt.and_utc().timestamp_nanos_opt(),
            TimeUnit::Microseconds => |dt| Some(datetime_to_timestamp_us(dt)),
            TimeUnit::Milliseconds => |dt| Some(datetime_to_timestamp_ms(dt)),
        };

        if tz_aware {
            #[cfg(feature = "timezones")]
            {
                let mut convert = LruCachedFunc::new(
                    |s: &str| {
                        let dt = DateTime::parse_from_str(s, &fmt).ok()?;
                        func(dt.naive_utc())
                    },
                    (string_ca.len() as f64).sqrt() as usize,
                );
                Ok(
                    unary_elementwise(string_ca, |opt_s| convert.eval(opt_s?, use_cache))
                        .with_name(string_ca.name().clone())
                        .into_datetime(tu, Some(tz.cloned().unwrap_or(TimeZone::UTC))),
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
            let ca = if strptime::fast_parser_supported(fmt.as_bytes()) {
                let mut strptime_cache = StrpTimeState::default();
                let mut convert = LruCachedFunc::new(
                    |s: &str| match strptime_cache.parse(s.as_bytes(), fmt.as_bytes()) {
                        None => transform(s, &fmt),
                        Some(ndt) => func(ndt),
                    },
                    (string_ca.len() as f64).sqrt() as usize,
                );
                unary_elementwise(string_ca, |opt_s| convert.eval(opt_s?, use_cache))
            } else {
                let mut convert = LruCachedFunc::new(
                    |s| transform(s, &fmt),
                    (string_ca.len() as f64).sqrt() as usize,
                );
                unary_elementwise(string_ca, |opt_s| convert.eval(opt_s?, use_cache))
            };
            let dt = ca
                .with_name(string_ca.name().clone())
                .into_datetime(tu, None);
            match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => crate::chunked_array::temporal::replace_time_zone::replace_time_zone(
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
