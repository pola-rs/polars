use arrow::array::PrimitiveArray;
use chrono::format::ParseErrorKind;
use chrono::{DateTime, NaiveDate, NaiveDateTime};
use once_cell::sync::Lazy;
use polars_core::prelude::*;
use regex::Regex;

use super::patterns::{self, Pattern};
#[cfg(feature = "dtype-date")]
use crate::chunkedarray::date::naive_date_to_date;
use crate::chunkedarray::string::strptime;
use crate::prelude::string::strptime::StrpTimeState;

const DATETIME_DMY_PATTERN: &str = r#"(?x)
        ^
        ['"]?                        # optional quotes
        (?:\d{1,2})                  # day
        [-/\.]                       # separator
        (?P<month>[01]?\d{1})        # month
        [-/\.]                       # separator
        (?:\d{4,})                   # year
        (?:
            [T\ ]                    # separator
            (?:\d{2})                # hour
            :?                       # separator
            (?:\d{2})                # minute
            (?:
                :?                   # separator
                (?:\d{2})            # second
                (?:
                    \.(?:\d{1,9})    # subsecond
                )?
            )?
        )?
        ['"]?                        # optional quotes
        $
        "#;

static DATETIME_DMY_RE: Lazy<Regex> = Lazy::new(|| Regex::new(DATETIME_DMY_PATTERN).unwrap());
const DATETIME_YMD_PATTERN: &str = r#"(?x)
        ^
        ['"]?                      # optional quotes
        (?:\d{4,})                 # year
        [-/\.]                     # separator
        (?P<month>[01]?\d{1})      # month
        [-/\.]                     # separator
        (?:\d{1,2})                # day
        (?:
            [T\ ]                  # separator
            (?:\d{2})              # hour
            :?                     # separator
            (?:\d{2})              # minute
            (?:
                :?                 # separator
                (?:\d{2})          # seconds
                (?:
                    \.(?:\d{1,9})  # subsecond
                )?
            )?
        )?
        ['"]?                      # optional quotes
        $
        "#;
static DATETIME_YMD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(DATETIME_YMD_PATTERN).unwrap());
const DATETIME_YMDZ_PATTERN: &str = r#"(?x)
        ^
        ['"]?                  # optional quotes
        (?:\d{4,})             # year
        [-/\.]                 # separator
        (?P<month>[01]?\d{1})  # month
        [-/\.]                 # separator
        (?:\d{1,2})            # year
        [T\ ]                  # separator
        (?:\d{2})              # hour
        :?                     # separator
        (?:\d{2})              # minute
        (?:
            :?                 # separator
            (?:\d{2})          # second
            (?:
                \.(?:\d{1,9})  # subsecond
            )?
        )?
        (?:
            # offset (e.g. +01:00)
            [+-](?:\d{2})
            :?
            (?:\d{2})
            # or Zulu suffix
            |Z
        )
        ['"]?                  # optional quotes
        $
        "#;
static DATETIME_YMDZ_RE: Lazy<Regex> = Lazy::new(|| Regex::new(DATETIME_YMDZ_PATTERN).unwrap());

impl Pattern {
    pub fn is_inferable(&self, val: &str) -> bool {
        match self {
            Pattern::DateDMY => true, // there are very few Date patterns, so it's cheaper
            Pattern::DateYMD => true, // to just try them
            Pattern::DatetimeDMY => match DATETIME_DMY_RE.captures(val) {
                Some(search) => (1..=12).contains(
                    &search
                        .name("month")
                        .unwrap()
                        .as_str()
                        .parse::<u8>()
                        .unwrap(),
                ),
                None => false,
            },
            Pattern::DatetimeYMD => match DATETIME_YMD_RE.captures(val) {
                Some(search) => (1..=12).contains(
                    &search
                        .name("month")
                        .unwrap()
                        .as_str()
                        .parse::<u8>()
                        .unwrap(),
                ),
                None => false,
            },
            Pattern::DatetimeYMDZ => match DATETIME_YMDZ_RE.captures(val) {
                Some(search) => (1..=12).contains(
                    &search
                        .name("month")
                        .unwrap()
                        .as_str()
                        .parse::<u8>()
                        .unwrap(),
                ),
                None => false,
            },
        }
    }
}

pub trait StrpTimeParser<T> {
    fn parse_bytes(&mut self, val: &[u8], time_unit: Option<TimeUnit>) -> Option<T>;
}

#[cfg(feature = "dtype-datetime")]
impl StrpTimeParser<i64> for DatetimeInfer<Int64Type> {
    fn parse_bytes(&mut self, val: &[u8], time_unit: Option<TimeUnit>) -> Option<i64> {
        if self.fmt_len == 0 {
            self.fmt_len = strptime::fmt_len(self.latest_fmt.as_bytes())?;
        }
        let transform = match time_unit {
            Some(TimeUnit::Nanoseconds) => datetime_to_timestamp_ns,
            Some(TimeUnit::Microseconds) => datetime_to_timestamp_us,
            Some(TimeUnit::Milliseconds) => datetime_to_timestamp_ms,
            _ => unreachable!(), // time_unit has to be provided for datetime
        };
        unsafe {
            self.transform_bytes
                .parse(val, self.latest_fmt.as_bytes(), self.fmt_len)
                .map(transform)
                .or_else(|| {
                    // TODO! this will try all patterns.
                    // somehow we must early escape if value is invalid
                    for fmt in self.patterns {
                        if self.fmt_len == 0 {
                            self.fmt_len = strptime::fmt_len(fmt.as_bytes())?;
                        }
                        if let Some(parsed) = self
                            .transform_bytes
                            .parse(val, fmt.as_bytes(), self.fmt_len)
                            .map(datetime_to_timestamp_us)
                        {
                            self.latest_fmt = fmt;
                            return Some(parsed);
                        }
                    }
                    None
                })
        }
    }
}

#[cfg(feature = "dtype-date")]
impl StrpTimeParser<i32> for DatetimeInfer<Int32Type> {
    fn parse_bytes(&mut self, val: &[u8], _time_unit: Option<TimeUnit>) -> Option<i32> {
        if self.fmt_len == 0 {
            self.fmt_len = strptime::fmt_len(self.latest_fmt.as_bytes())?;
        }
        unsafe {
            self.transform_bytes
                .parse(val, self.latest_fmt.as_bytes(), self.fmt_len)
                .map(|ndt| naive_date_to_date(ndt.date()))
                .or_else(|| {
                    // TODO! this will try all patterns.
                    // somehow we must early escape if value is invalid
                    for fmt in self.patterns {
                        if self.fmt_len == 0 {
                            self.fmt_len = strptime::fmt_len(fmt.as_bytes())?;
                        }
                        if let Some(parsed) = self
                            .transform_bytes
                            .parse(val, fmt.as_bytes(), self.fmt_len)
                            .map(|ndt| naive_date_to_date(ndt.date()))
                        {
                            self.latest_fmt = fmt;
                            return Some(parsed);
                        }
                    }
                    None
                })
        }
    }
}

#[derive(Clone)]
pub struct DatetimeInfer<T: PolarsNumericType> {
    pub pattern: Pattern,
    patterns: &'static [&'static str],
    latest_fmt: &'static str,
    transform: fn(&str, &str) -> Option<T::Native>,
    transform_bytes: StrpTimeState,
    fmt_len: u16,
    pub logical_type: DataType,
}

pub trait TryFromWithUnit<T>: Sized {
    type Error;
    fn try_from_with_unit(pattern: T, unit: Option<TimeUnit>) -> PolarsResult<Self>;
}

#[cfg(feature = "dtype-datetime")]
impl TryFromWithUnit<Pattern> for DatetimeInfer<Int64Type> {
    type Error = PolarsError;

    fn try_from_with_unit(value: Pattern, time_unit: Option<TimeUnit>) -> PolarsResult<Self> {
        let time_unit = time_unit.expect("time_unit must be provided for datetime");

        let transform = match (time_unit, value) {
            (TimeUnit::Milliseconds, Pattern::DatetimeYMDZ) => transform_tzaware_datetime_ms,
            (TimeUnit::Milliseconds, _) => transform_datetime_ms,
            (TimeUnit::Microseconds, Pattern::DatetimeYMDZ) => transform_tzaware_datetime_us,
            (TimeUnit::Microseconds, _) => transform_datetime_us,
            (TimeUnit::Nanoseconds, Pattern::DatetimeYMDZ) => transform_tzaware_datetime_ns,
            (TimeUnit::Nanoseconds, _) => transform_datetime_ns,
        };
        let (pattern, patterns) = match value {
            Pattern::DatetimeDMY | Pattern::DateDMY => {
                (Pattern::DatetimeDMY, patterns::DATETIME_D_M_Y)
            },
            Pattern::DatetimeYMD | Pattern::DateYMD => {
                (Pattern::DatetimeYMD, patterns::DATETIME_Y_M_D)
            },
            Pattern::DatetimeYMDZ => (Pattern::DatetimeYMDZ, patterns::DATETIME_Y_M_D_Z),
        };

        Ok(DatetimeInfer {
            pattern,
            patterns,
            latest_fmt: patterns[0],
            transform,
            transform_bytes: StrpTimeState::default(),
            fmt_len: 0,
            logical_type: DataType::Datetime(time_unit, None),
        })
    }
}

#[cfg(feature = "dtype-date")]
impl TryFromWithUnit<Pattern> for DatetimeInfer<Int32Type> {
    type Error = PolarsError;

    fn try_from_with_unit(value: Pattern, _time_unit: Option<TimeUnit>) -> PolarsResult<Self> {
        match value {
            Pattern::DateDMY => Ok(DatetimeInfer {
                pattern: Pattern::DateDMY,
                patterns: patterns::DATE_D_M_Y,
                latest_fmt: patterns::DATE_D_M_Y[0],
                transform: transform_date,
                transform_bytes: StrpTimeState::default(),
                fmt_len: 0,
                logical_type: DataType::Date,
            }),
            Pattern::DateYMD => Ok(DatetimeInfer {
                pattern: Pattern::DateYMD,
                patterns: patterns::DATE_Y_M_D,
                latest_fmt: patterns::DATE_Y_M_D[0],
                transform: transform_date,
                transform_bytes: StrpTimeState::default(),
                fmt_len: 0,
                logical_type: DataType::Date,
            }),
            _ => polars_bail!(ComputeError: "could not convert pattern"),
        }
    }
}

impl<T: PolarsNumericType> DatetimeInfer<T> {
    pub fn parse(&mut self, val: &str) -> Option<T::Native> {
        match (self.transform)(val, self.latest_fmt) {
            Some(parsed) => Some(parsed),
            // try other patterns
            None => {
                if !self.pattern.is_inferable(val) {
                    return None;
                }
                for fmt in self.patterns {
                    self.fmt_len = 0;
                    if let Some(parsed) = (self.transform)(val, fmt) {
                        self.latest_fmt = fmt;
                        return Some(parsed);
                    }
                }
                None
            },
        }
    }
}

impl<T: PolarsNumericType> DatetimeInfer<T>
where
    ChunkedArray<T>: IntoSeries,
{
    fn coerce_string(&mut self, ca: &StringChunked) -> Series {
        let chunks = ca.downcast_iter().map(|array| {
            let iter = array
                .into_iter()
                .map(|opt_val| opt_val.and_then(|val| self.parse(val)));
            PrimitiveArray::from_trusted_len_iter(iter)
        });
        ChunkedArray::from_chunk_iter(ca.name(), chunks)
            .into_series()
            .cast(&self.logical_type)
            .unwrap()
            .with_name(ca.name())
    }
}

#[cfg(feature = "dtype-date")]
fn transform_date(val: &str, fmt: &str) -> Option<i32> {
    NaiveDate::parse_from_str(val, fmt)
        .ok()
        .map(naive_date_to_date)
}

#[cfg(feature = "dtype-datetime")]
pub(crate) fn transform_datetime_ns(val: &str, fmt: &str) -> Option<i64> {
    match NaiveDateTime::parse_from_str(val, fmt) {
        Ok(ndt) => Some(datetime_to_timestamp_ns(ndt)),
        Err(parse_error) => match parse_error.kind() {
            ParseErrorKind::NotEnough => NaiveDate::parse_from_str(val, fmt)
                .ok()
                .map(|nd| datetime_to_timestamp_ns(nd.and_hms_opt(0, 0, 0).unwrap())),
            _ => None,
        },
    }
}

fn transform_tzaware_datetime_ns(val: &str, fmt: &str) -> Option<i64> {
    let dt = DateTime::parse_from_str(val, fmt);
    dt.ok().map(|dt| datetime_to_timestamp_ns(dt.naive_utc()))
}

#[cfg(feature = "dtype-datetime")]
pub(crate) fn transform_datetime_us(val: &str, fmt: &str) -> Option<i64> {
    match NaiveDateTime::parse_from_str(val, fmt) {
        Ok(ndt) => Some(datetime_to_timestamp_us(ndt)),
        Err(parse_error) => match parse_error.kind() {
            ParseErrorKind::NotEnough => NaiveDate::parse_from_str(val, fmt)
                .ok()
                .map(|nd| datetime_to_timestamp_us(nd.and_hms_opt(0, 0, 0).unwrap())),
            _ => None,
        },
    }
}

fn transform_tzaware_datetime_us(val: &str, fmt: &str) -> Option<i64> {
    let dt = DateTime::parse_from_str(val, fmt);
    dt.ok().map(|dt| datetime_to_timestamp_us(dt.naive_utc()))
}

#[cfg(feature = "dtype-datetime")]
pub(crate) fn transform_datetime_ms(val: &str, fmt: &str) -> Option<i64> {
    match NaiveDateTime::parse_from_str(val, fmt) {
        Ok(ndt) => Some(datetime_to_timestamp_ms(ndt)),
        Err(parse_error) => match parse_error.kind() {
            ParseErrorKind::NotEnough => NaiveDate::parse_from_str(val, fmt)
                .ok()
                .map(|nd| datetime_to_timestamp_ms(nd.and_hms_opt(0, 0, 0).unwrap())),
            _ => None,
        },
    }
}

fn transform_tzaware_datetime_ms(val: &str, fmt: &str) -> Option<i64> {
    let dt = DateTime::parse_from_str(val, fmt);
    dt.ok().map(|dt| datetime_to_timestamp_ms(dt.naive_utc()))
}

pub fn infer_pattern_single(val: &str) -> Option<Pattern> {
    // Dates come first, because we see datetimes as superset of dates
    infer_pattern_date_single(val).or_else(|| infer_pattern_datetime_single(val))
}

fn infer_pattern_datetime_single(val: &str) -> Option<Pattern> {
    if patterns::DATETIME_D_M_Y.iter().any(|fmt| {
        NaiveDateTime::parse_from_str(val, fmt).is_ok()
            || NaiveDate::parse_from_str(val, fmt).is_ok()
    }) {
        Some(Pattern::DatetimeDMY)
    } else if patterns::DATETIME_Y_M_D.iter().any(|fmt| {
        NaiveDateTime::parse_from_str(val, fmt).is_ok()
            || NaiveDate::parse_from_str(val, fmt).is_ok()
    }) {
        Some(Pattern::DatetimeYMD)
    } else if patterns::DATETIME_Y_M_D_Z
        .iter()
        .any(|fmt| NaiveDateTime::parse_from_str(val, fmt).is_ok())
    {
        Some(Pattern::DatetimeYMDZ)
    } else {
        None
    }
}

fn infer_pattern_date_single(val: &str) -> Option<Pattern> {
    if patterns::DATE_D_M_Y
        .iter()
        .any(|fmt| NaiveDate::parse_from_str(val, fmt).is_ok())
    {
        Some(Pattern::DateDMY)
    } else if patterns::DATE_Y_M_D
        .iter()
        .any(|fmt| NaiveDate::parse_from_str(val, fmt).is_ok())
    {
        Some(Pattern::DateYMD)
    } else {
        None
    }
}

#[cfg(feature = "dtype-datetime")]
pub(crate) fn to_datetime(
    ca: &StringChunked,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
    _ambiguous: &StringChunked,
) -> PolarsResult<DatetimeChunked> {
    match ca.first_non_null() {
        None => Ok(Int64Chunked::full_null(ca.name(), ca.len()).into_datetime(tu, tz.cloned())),
        Some(idx) => {
            let subset = ca.slice(idx as i64, ca.len());
            let pattern = subset
                .into_iter()
                .find_map(|opt_val| opt_val.and_then(infer_pattern_datetime_single))
                .ok_or_else(|| polars_err!(parse_fmt_idk = "date"))?;
            let mut infer = DatetimeInfer::<Int64Type>::try_from_with_unit(pattern, Some(tu))?;
            match pattern {
                #[cfg(feature = "timezones")]
                Pattern::DatetimeYMDZ => infer.coerce_string(ca).datetime().map(|ca| {
                    let mut ca = ca.clone();
                    // `tz` has already been validated.
                    ca.set_time_unit_and_time_zone(
                        tu,
                        tz.cloned().unwrap_or_else(|| "UTC".to_string()),
                    )?;
                    Ok(ca)
                })?,
                _ => infer.coerce_string(ca).datetime().map(|ca| {
                    let mut ca = ca.clone();
                    ca.set_time_unit(tu);
                    match tz {
                        #[cfg(feature = "timezones")]
                        Some(tz) => polars_ops::prelude::replace_time_zone(
                            &ca,
                            Some(tz),
                            _ambiguous,
                            NonExistent::Raise,
                        ),
                        _ => Ok(ca),
                    }
                })?,
            }
        },
    }
}
#[cfg(feature = "dtype-date")]
pub(crate) fn to_date(ca: &StringChunked) -> PolarsResult<DateChunked> {
    match ca.first_non_null() {
        None => Ok(Int32Chunked::full_null(ca.name(), ca.len()).into_date()),
        Some(idx) => {
            let subset = ca.slice(idx as i64, ca.len());
            let pattern = subset
                .into_iter()
                .find_map(|opt_val| opt_val.and_then(infer_pattern_date_single))
                .ok_or_else(|| polars_err!(parse_fmt_idk = "date"))?;
            let mut infer = DatetimeInfer::<Int32Type>::try_from_with_unit(pattern, None).unwrap();
            infer.coerce_string(ca).date().cloned()
        },
    }
}
