use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime};
use once_cell::sync::Lazy;
use polars_arrow::export::arrow::array::PrimitiveArray;
use polars_core::prelude::*;
use polars_core::utils::arrow::types::NativeType;
use regex::Regex;

use super::patterns::{self, PatternWithOffset};
#[cfg(feature = "dtype-date")]
use crate::chunkedarray::date::naive_date_to_date;
use crate::chunkedarray::utf8::patterns::Pattern;
use crate::chunkedarray::utf8::strptime;
use crate::prelude::utf8::strptime::StrpTimeState;

const DATETIME_DMY_PATTERN: &str = r#"(?x)
        ^
        ['"]?                        # optional quotes
        (?:\d{1,2})                  # day
        [-/]                         # separator
        (?P<month>[01]?\d{1})        # month
        [-/]                         # separator
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
        [-/]                       # separator
        (?P<month>[01]?\d{1})      # month
        [-/]                       # separator
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
        [-/]                   # separator
        (?P<month>[01]?\d{1})  # month
        [-/]                   # separator
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
    fn parse_bytes(&mut self, val: &[u8]) -> Option<T>;
}

#[cfg(feature = "dtype-datetime")]
impl StrpTimeParser<i64> for DatetimeInfer<i64> {
    fn parse_bytes(&mut self, val: &[u8]) -> Option<i64> {
        if self.fmt_len == 0 {
            self.fmt_len = strptime::fmt_len(self.latest_fmt.as_bytes())?;
        }
        unsafe {
            self.transform_bytes
                .parse(val, self.latest_fmt.as_bytes(), self.fmt_len, true)
                .map(datetime_to_timestamp_us)
                .or_else(|| {
                    // TODO! this will try all patterns.
                    // somehow we must early escape if value is invalid
                    for fmt in self.patterns {
                        if self.fmt_len == 0 {
                            self.fmt_len = strptime::fmt_len(fmt.as_bytes())?;
                        }
                        if let Some(parsed) = self
                            .transform_bytes
                            .parse(val, fmt.as_bytes(), self.fmt_len, true)
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
impl StrpTimeParser<i32> for DatetimeInfer<i32> {
    fn parse_bytes(&mut self, val: &[u8]) -> Option<i32> {
        if self.fmt_len == 0 {
            self.fmt_len = strptime::fmt_len(self.latest_fmt.as_bytes())?;
        }
        unsafe {
            self.transform_bytes
                .parse(val, self.latest_fmt.as_bytes(), self.fmt_len, false)
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
                            .parse(val, fmt.as_bytes(), self.fmt_len, false)
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
pub struct DatetimeInfer<T> {
    pub pattern_with_offset: PatternWithOffset,
    patterns: &'static [&'static str],
    latest_fmt: &'static str,
    transform: fn(&str, &str, Option<FixedOffset>, bool) -> Option<T>,
    transform_bytes: StrpTimeState,
    fmt_len: u16,
    pub logical_type: DataType,
    utc: bool,
}

#[cfg(feature = "dtype-datetime")]
impl TryFrom<Pattern> for DatetimeInfer<i64> {
    type Error = PolarsError;

    fn try_from(value: Pattern) -> PolarsResult<Self> {
        match value {
            Pattern::DatetimeDMY => Ok(DatetimeInfer {
                pattern_with_offset: PatternWithOffset {
                    pattern: Pattern::DatetimeDMY,
                    offset: None,
                },
                patterns: patterns::DATETIME_D_M_Y,
                latest_fmt: patterns::DATETIME_D_M_Y[0],
                transform: transform_datetime_us,
                transform_bytes: StrpTimeState::default(),
                fmt_len: 0,
                logical_type: DataType::Datetime(TimeUnit::Microseconds, None),
                utc: false,
            }),
            Pattern::DatetimeYMD => Ok(DatetimeInfer {
                pattern_with_offset: PatternWithOffset {
                    pattern: Pattern::DatetimeYMD,
                    offset: None,
                },
                patterns: patterns::DATETIME_Y_M_D,
                latest_fmt: patterns::DATETIME_Y_M_D[0],
                transform: transform_datetime_us,
                transform_bytes: StrpTimeState::default(),
                fmt_len: 0,
                logical_type: DataType::Datetime(TimeUnit::Microseconds, None),
                utc: false,
            }),
            Pattern::DatetimeYMDZ => Ok(DatetimeInfer {
                pattern_with_offset: PatternWithOffset {
                    pattern: Pattern::DatetimeYMDZ,
                    offset: None,
                },
                patterns: patterns::DATETIME_Y_M_D_Z,
                latest_fmt: patterns::DATETIME_Y_M_D_Z[0],
                transform: transform_tzaware_datetime_us,
                transform_bytes: StrpTimeState::default(),
                fmt_len: 0,
                logical_type: DataType::Datetime(TimeUnit::Microseconds, None),
                utc: false,
            }),
            _ => polars_bail!(ComputeError: "could not convert pattern"),
        }
    }
}

#[cfg(feature = "dtype-date")]
impl TryFrom<Pattern> for DatetimeInfer<i32> {
    type Error = PolarsError;

    fn try_from(value: Pattern) -> PolarsResult<Self> {
        match value {
            Pattern::DateDMY => Ok(DatetimeInfer {
                pattern_with_offset: PatternWithOffset {
                    pattern: Pattern::DateDMY,
                    offset: None,
                },
                patterns: patterns::DATE_D_M_Y,
                latest_fmt: patterns::DATE_D_M_Y[0],
                transform: transform_date,
                transform_bytes: StrpTimeState::default(),
                fmt_len: 0,
                logical_type: DataType::Date,
                utc: false,
            }),
            Pattern::DateYMD => Ok(DatetimeInfer {
                pattern_with_offset: PatternWithOffset {
                    pattern: Pattern::DateYMD,
                    offset: None,
                },
                patterns: patterns::DATE_Y_M_D,
                latest_fmt: patterns::DATE_Y_M_D[0],
                transform: transform_date,
                transform_bytes: StrpTimeState::default(),
                fmt_len: 0,
                logical_type: DataType::Date,
                utc: false,
            }),
            _ => polars_bail!(ComputeError: "could not convert pattern"),
        }
    }
}

impl<T: NativeType> DatetimeInfer<T> {
    pub fn parse(&mut self, val: &str, offset: Option<FixedOffset>) -> Option<T> {
        match (self.transform)(val, self.latest_fmt, offset, self.utc) {
            Some(parsed) => Some(parsed),
            // try other patterns
            None => {
                if !self.pattern_with_offset.pattern.is_inferable(val) {
                    return None;
                }
                for fmt in self.patterns {
                    self.fmt_len = 0;
                    if let Some(parsed) = (self.transform)(val, fmt, offset, self.utc) {
                        self.latest_fmt = fmt;
                        return Some(parsed);
                    }
                }
                None
            }
        }
    }

    fn coerce_utf8(&mut self, ca: &Utf8Chunked, offset: Option<FixedOffset>) -> Series {
        let chunks = ca
            .downcast_iter()
            .map(|array| {
                let iter = array
                    .into_iter()
                    .map(|opt_val| opt_val.and_then(|val| self.parse(val, offset)));
                Box::new(PrimitiveArray::from_trusted_len_iter(iter)) as ArrayRef
            })
            .collect();
        let mut out = match self.logical_type {
            DataType::Date => unsafe { Int32Chunked::from_chunks(ca.name(), chunks) }
                .into_series()
                .cast(&self.logical_type)
                .unwrap(),
            DataType::Datetime(_, _) => unsafe { Int64Chunked::from_chunks(ca.name(), chunks) }
                .into_series()
                .cast(&self.logical_type)
                .unwrap(),
            _ => unreachable!(),
        };
        out.rename(ca.name());
        out
    }
}

#[cfg(feature = "dtype-date")]
fn transform_date(val: &str, fmt: &str, _offset: Option<FixedOffset>, _utc: bool) -> Option<i32> {
    NaiveDate::parse_from_str(val, fmt)
        .ok()
        .map(naive_date_to_date)
}

#[cfg(feature = "dtype-datetime")]
pub(crate) fn transform_datetime_ns(
    val: &str,
    fmt: &str,
    _offset: Option<FixedOffset>,
    _utc: bool,
) -> Option<i64> {
    NaiveDateTime::parse_from_str(val, fmt)
        .ok()
        .map(datetime_to_timestamp_ns)
}

fn transform_tzaware_datetime_ns(
    val: &str,
    fmt: &str,
    offset: Option<FixedOffset>,
    utc: bool,
) -> Option<i64> {
    let dt = DateTime::parse_from_str(val, fmt);
    match utc {
        true => dt.ok().map(|dt| datetime_to_timestamp_ns(dt.naive_utc())),
        false => match Some(dt.ok()?.timezone()) == offset {
            true => dt.ok().map(|dt| datetime_to_timestamp_ns(dt.naive_utc())),
            false => None,
        },
    }
}

#[cfg(feature = "dtype-datetime")]
pub(crate) fn transform_datetime_us(
    val: &str,
    fmt: &str,
    _offset: Option<FixedOffset>,
    _utc: bool,
) -> Option<i64> {
    NaiveDateTime::parse_from_str(val, fmt)
        .ok()
        .map(datetime_to_timestamp_us)
}

fn transform_tzaware_datetime_us(
    val: &str,
    fmt: &str,
    offset: Option<FixedOffset>,
    utc: bool,
) -> Option<i64> {
    let dt = DateTime::parse_from_str(val, fmt);
    match utc {
        true => dt.ok().map(|dt| datetime_to_timestamp_us(dt.naive_utc())),
        false => match Some(dt.ok()?.timezone()) == offset {
            true => dt.ok().map(|dt| datetime_to_timestamp_us(dt.naive_utc())),
            false => None,
        },
    }
}

#[cfg(feature = "dtype-datetime")]
pub(crate) fn transform_datetime_ms(
    val: &str,
    fmt: &str,
    _offset: Option<FixedOffset>,
    _utc: bool,
) -> Option<i64> {
    NaiveDateTime::parse_from_str(val, fmt)
        .ok()
        .map(datetime_to_timestamp_ms)
}

fn transform_tzaware_datetime_ms(
    val: &str,
    fmt: &str,
    offset: Option<FixedOffset>,
    utc: bool,
) -> Option<i64> {
    let dt = DateTime::parse_from_str(val, fmt);
    match utc {
        true => dt.ok().map(|dt| datetime_to_timestamp_ms(dt.naive_utc())),
        false => match Some(dt.ok()?.timezone()) == offset {
            true => dt.ok().map(|dt| datetime_to_timestamp_ms(dt.naive_utc())),
            false => None,
        },
    }
}

pub fn infer_pattern_single(val: &str) -> Option<PatternWithOffset> {
    // Dates come first, because we see datetimes as superset of dates
    infer_pattern_date_single(val).or_else(|| infer_pattern_datetime_single(val))
}

fn infer_pattern_datetime_single(val: &str) -> Option<PatternWithOffset> {
    if patterns::DATETIME_D_M_Y.iter().any(|fmt| {
        NaiveDateTime::parse_from_str(val, fmt).is_ok()
            || NaiveDate::parse_from_str(val, fmt).is_ok()
    }) {
        Some(PatternWithOffset {
            pattern: Pattern::DatetimeDMY,
            offset: None,
        })
    } else if patterns::DATETIME_Y_M_D.iter().any(|fmt| {
        NaiveDateTime::parse_from_str(val, fmt).is_ok()
            || NaiveDate::parse_from_str(val, fmt).is_ok()
    }) {
        Some(PatternWithOffset {
            pattern: Pattern::DatetimeYMD,
            offset: None,
        })
    } else {
        patterns::DATETIME_Y_M_D_Z
            .iter()
            .find_map(|fmt| DateTime::parse_from_str(val, fmt).ok())
            .map(|dt| PatternWithOffset {
                pattern: Pattern::DatetimeYMDZ,
                offset: Some(dt.timezone()),
            })
    }
}

fn infer_pattern_date_single(val: &str) -> Option<PatternWithOffset> {
    if patterns::DATE_D_M_Y
        .iter()
        .any(|fmt| NaiveDate::parse_from_str(val, fmt).is_ok())
    {
        Some(PatternWithOffset {
            pattern: Pattern::DateDMY,
            offset: None,
        })
    } else if patterns::DATE_Y_M_D
        .iter()
        .any(|fmt| NaiveDate::parse_from_str(val, fmt).is_ok())
    {
        Some(PatternWithOffset {
            pattern: Pattern::DateYMD,
            offset: None,
        })
    } else {
        None
    }
}

#[cfg(feature = "dtype-datetime")]
pub(crate) fn to_datetime(
    ca: &Utf8Chunked,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
    utc: bool,
) -> PolarsResult<DatetimeChunked> {
    match ca.first_non_null() {
        None => Ok(Int64Chunked::full_null(ca.name(), ca.len()).into_datetime(tu, tz.cloned())),
        Some(idx) => {
            let subset = ca.slice(idx as i64, ca.len());
            let pattern_with_offset = subset
                .into_iter()
                .find_map(|opt_val| opt_val.and_then(infer_pattern_datetime_single))
                .ok_or_else(|| polars_err!(parse_fmt_idk = "date"))?;
            let mut infer = DatetimeInfer::<i64>::try_from(pattern_with_offset.pattern)?;
            match (tu, pattern_with_offset.offset) {
                (TimeUnit::Nanoseconds, None) => infer.transform = transform_datetime_ns,
                (TimeUnit::Microseconds, None) => infer.transform = transform_datetime_us,
                (TimeUnit::Milliseconds, None) => infer.transform = transform_datetime_ms,
                (TimeUnit::Nanoseconds, _) => infer.transform = transform_tzaware_datetime_ns,
                (TimeUnit::Microseconds, _) => infer.transform = transform_tzaware_datetime_us,
                (TimeUnit::Milliseconds, _) => infer.transform = transform_tzaware_datetime_ms,
            }
            infer.utc = utc;
            if tz.is_some() && pattern_with_offset.offset.is_some() {
                polars_bail!(ComputeError: "cannot parse tz-aware values with tz-aware dtype - please drop the time zone from the dtype.")
            }
            match pattern_with_offset.offset {
                #[cfg(feature = "timezones")]
                Some(offset) => infer.coerce_utf8(ca, Some(offset)).datetime().map(|ca| {
                    let mut ca = ca.clone();
                    ca.set_time_unit(tu);
                    match utc {
                        true => Ok(ca.replace_time_zone(Some("UTC"), None)?),
                        false => Ok(ca
                            .replace_time_zone(Some("UTC"), None)?
                            .convert_time_zone(offset.to_string())?),
                    }
                })?,
                _ => infer.coerce_utf8(ca, None).datetime().map(|ca| {
                    let mut ca = ca.clone();
                    ca.set_time_unit(tu);
                    match (tz, utc) {
                        #[cfg(feature = "timezones")]
                        (Some(tz), false) => Ok(ca.replace_time_zone(Some(tz), None)?),
                        #[cfg(feature = "timezones")]
                        (None, true) => Ok(ca.replace_time_zone(Some("UTC"), None)?),
                        #[cfg(feature = "timezones")]
                        (Some(_), true) => unreachable!(), // has already been validated in strptime
                        _ => Ok(ca),
                    }
                })?,
            }
        }
    }
}
#[cfg(feature = "dtype-date")]
pub(crate) fn to_date(ca: &Utf8Chunked) -> PolarsResult<DateChunked> {
    match ca.first_non_null() {
        None => Ok(Int32Chunked::full_null(ca.name(), ca.len()).into_date()),
        Some(idx) => {
            let subset = ca.slice(idx as i64, ca.len());
            let pattern_with_offset = subset
                .into_iter()
                .find_map(|opt_val| opt_val.and_then(infer_pattern_date_single))
                .ok_or_else(|| polars_err!(parse_fmt_idk = "date"))?;
            let mut infer = DatetimeInfer::<i32>::try_from(pattern_with_offset.pattern).unwrap();
            infer.coerce_utf8(ca, None).date().cloned()
        }
    }
}
