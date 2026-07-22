use arrow::array::PrimitiveArray;
use jiff::Timestamp;
use jiff::civil::{Date as NaiveDate, DateTime as NaiveDateTime, Time as NaiveTime};
use jiff::fmt::strtime::BrokenDownTime;
use polars_core::prelude::*;

use super::patterns::{self, Pattern};
#[cfg(feature = "dtype-date")]
use crate::chunkedarray::date::naive_date_to_date;
use crate::prelude::string::strptime::StrpTimeState;

polars_utils::regex_cache::cached_regex! {
    // A `%f`/`%.f` fractional-seconds directive with an explicit, fixed
    // digit-count precision, e.g. `%3f`, `%.6f`, `%9f`.
    static EXPLICIT_FRACTIONAL_PRECISION_RE = r"%\.?([1-9])f";

    static DATETIME_DMY_RE = r#"(?x)
        ^
        ['"]?                        # optional quotes
        (?:\d{1,2})                  # day
        [-/\.]                       # separator
        (?P<month>[01]?\d{1})        # month
        [-/\.]                       # separator
        (?:\d{4,})                   # year
        (?:
            [T\ ]                    # separator
            (?:\d{1,2})              # hour
            :?                       # separator
            (?:\d{1,2})              # minute
            (?:
                :?                   # separator
                (?:\d{1,2})          # second
                (?:
                    \.(?:\d{1,9})    # subsecond
                )?
            )?
        )?
        ['"]?                        # optional quotes
        $
        "#;

    static DATETIME_YMD_RE = r#"(?x)
            ^
            ['"]?                      # optional quotes
            (?:\d{4,})                 # year
            [-/\.]?                    # separator
            (?P<month>[01]?\d{1})      # month
            [-/\.]?                    # separator
            (?:\d{1,2})                # day
            (?:
                [T\ ]                  # separator
                (?:\d{1,2})            # hour
                :?                     # separator
                (?:\d{1,2})            # minute
                (?:
                    :?                 # separator
                    (?:\d{1,2})        # seconds
                    (?:
                        \.(?:\d{1,9})  # subsecond
                    )?
                )?
            )?
            ['"]?                      # optional quotes
            $
            "#;

    static DATETIME_YMDZ_RE = r#"(?x)
            ^
            ['"]?                  # optional quotes
            (?:\d{4,})             # year
            [-/\.]?                # separator
            (?P<month>[01]?\d{1})  # month
            [-/\.]?                # separator
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
                # offset (e.g. +01:00, +0100, or +01)
                [+-](?:\d{2})
                (?::?\d{2})?
                # or Zulu suffix
                |Z
            )
            ['"]?                  # optional quotes
            $
            "#;
}

impl Pattern {
    pub fn is_inferable(&self, val: &str) -> bool {
        match self {
            Pattern::DateDMY => true, // there are very few Date patterns, so it's cheaper
            Pattern::DateYMD => true, // to just try them
            Pattern::Time => true,
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
        let transform = match time_unit {
            Some(TimeUnit::Nanoseconds) => datetime_to_timestamp_ns,
            Some(TimeUnit::Microseconds) => datetime_to_timestamp_us,
            Some(TimeUnit::Milliseconds) => datetime_to_timestamp_ms,
            _ => unreachable!(), // time_unit has to be provided for datetime
        };
        self.transform_bytes
            .parse(val, self.latest_fmt.as_bytes())
            .map(transform)
            .or_else(|| {
                // TODO! this will try all patterns.
                // Somehow we must early escape if value is invalid.
                for fmt in self.patterns {
                    if let Some(parsed) = self
                        .transform_bytes
                        .parse(val, fmt.as_bytes())
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

#[cfg(feature = "dtype-date")]
impl StrpTimeParser<i32> for DatetimeInfer<Int32Type> {
    fn parse_bytes(&mut self, val: &[u8], _time_unit: Option<TimeUnit>) -> Option<i32> {
        self.transform_bytes
            .parse(val, self.latest_fmt.as_bytes())
            .map(|ndt| naive_date_to_date(ndt.date()))
            .or_else(|| {
                // TODO! this will try all patterns.
                // somehow we must early escape if value is invalid
                for fmt in self.patterns {
                    if let Some(parsed) = self
                        .transform_bytes
                        .parse(val, fmt.as_bytes())
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

#[derive(Clone)]
pub struct DatetimeInfer<T: PolarsNumericType> {
    pub pattern: Pattern,
    patterns: &'static [&'static str],
    latest_fmt: &'static str,
    transform: fn(&str, &str) -> Option<T::Native>,
    transform_bytes: StrpTimeState,
    pub logical_type: DataType,
}

pub trait TryFromWithUnit<T>: Sized {
    type Error;
    fn try_from_with_unit(pattern: T, time_unit: Option<TimeUnit>) -> PolarsResult<Self>;
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
            Pattern::Time => (Pattern::Time, patterns::TIME_H_M_S),
        };

        Ok(DatetimeInfer {
            pattern,
            patterns,
            latest_fmt: patterns[0],
            transform,
            transform_bytes: StrpTimeState::default(),
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
                logical_type: DataType::Date,
            }),
            Pattern::DateYMD => Ok(DatetimeInfer {
                pattern: Pattern::DateYMD,
                patterns: patterns::DATE_Y_M_D,
                latest_fmt: patterns::DATE_Y_M_D[0],
                transform: transform_date,
                transform_bytes: StrpTimeState::default(),
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

impl<T: PolarsNumericType> DatetimeInfer<T> {
    pub fn coerce_string(&mut self, ca: &StringChunked) -> Series {
        let chunks = ca.downcast_iter().map(|array| {
            let iter = array
                .into_iter()
                .map(|opt_val| opt_val.and_then(|val| self.parse(val)));
            PrimitiveArray::from_trusted_len_iter(iter)
        });
        ChunkedArray::<T>::from_chunk_iter(ca.name().clone(), chunks)
            .into_series()
            .cast(&self.logical_type)
            .unwrap()
            .with_name(ca.name().clone())
    }
}

#[cfg(feature = "dtype-date")]
fn transform_date(val: &str, fmt: &str) -> Option<i32> {
    NaiveDate::strptime(fmt, val).ok().map(naive_date_to_date)
}

pub(crate) fn parse_datetime(val: &str, fmt: &str) -> Option<NaiveDateTime> {
    let dt = NaiveDateTime::strptime(fmt, val)
        .ok()
        .or_else(|| {
            // Fall back to a date-only parse (e.g. the format has no time component).
            NaiveDate::strptime(fmt, val)
                .ok()
                .map(|nd| nd.at(0, 0, 0, 0))
        })
        .or_else(|| {
            // jiff refuses `%Z` (a time zone abbreviation, e.g. "ACST")
            // outright when parsing - it's formatting-only, since an
            // abbreviation alone is ambiguous. Chrono allowed it as a
            // "match and discard" escape hatch: whatever it matched was
            // just thrown away, so replicate that by stripping the
            // directive and consuming a trailing run of letters in its
            // place.
            let fmt_prefix = fmt.strip_suffix("%Z")?;
            let (tm, len) = BrokenDownTime::parse_prefix(fmt_prefix, val).ok()?;
            let remainder = &val[len..];
            if remainder.is_empty() || !remainder.bytes().all(|b| b.is_ascii_alphabetic()) {
                return None;
            }
            tm.to_datetime().ok()
        })?;

    // jiff treats the digit count in `%3f`/`%.6f`/etc. as formatting-only -
    // during parsing it's a no-op, consuming up to nanosecond precision
    // regardless of what's requested. Restore the old, stricter behavior of
    // failing when the data actually carries more precision than the
    // format explicitly promised (e.g. `%.3f` against a value with
    // nanosecond-level digits), rather than silently truncating it.
    if let Some(declared) = EXPLICIT_FRACTIONAL_PRECISION_RE
        .captures(fmt)
        .and_then(|c| c.get(1)?.as_str().parse::<u32>().ok())
    {
        let subsec = dt.subsec_nanosecond() as u32;
        if declared < 9 && subsec % 10u32.pow(9 - declared) != 0 {
            return None;
        }
    }

    Some(dt)
}

pub(crate) fn parse_datetime_and_remainder<'a>(
    val: &'a str,
    fmt: &str,
) -> Option<(NaiveDateTime, &'a str)> {
    BrokenDownTime::parse_prefix(fmt, val)
        .ok()
        .and_then(|(tm, len)| tm.to_datetime().ok().map(|dt| (dt, &val[len..])))
        .or_else(|| {
            // Fall back to a date-only parse (e.g. the format has no time component).
            BrokenDownTime::parse_prefix(fmt, val)
                .ok()
                .and_then(|(tm, len)| tm.to_date().ok().map(|nd| (nd.at(0, 0, 0, 0), &val[len..])))
        })
}

#[cfg(feature = "dtype-datetime")]
pub(crate) fn transform_datetime_ns(val: &str, fmt: &str) -> Option<i64> {
    parse_datetime(val, fmt).map(datetime_to_timestamp_ns)
}

#[cfg(feature = "dtype-datetime")]
pub(crate) fn transform_datetime_us(val: &str, fmt: &str) -> Option<i64> {
    parse_datetime(val, fmt).map(datetime_to_timestamp_us)
}

#[cfg(feature = "dtype-datetime")]
pub(crate) fn transform_datetime_ms(val: &str, fmt: &str) -> Option<i64> {
    parse_datetime(val, fmt).map(datetime_to_timestamp_ms)
}

/// Parses a tz-aware datetime string, honoring whatever offset (or "Z") is
/// embedded in the string. Unlike the naive parsers, every pattern in this
/// module's tz-aware pattern list is expected to carry offset/zone info; if
/// none can be extracted, the value is treated as UTC.
fn parse_tz_aware_timestamp(val: &str, fmt: &str) -> Option<Timestamp> {
    if fmt == "%+" {
        // "%+" mirrors chrono's combined ISO 8601 / RFC 3339 format specifier,
        // which jiff's strtime engine does not implement as a directive; jiff's
        // native ISO 8601 timestamp parser handles the same cases directly,
        // including a literal "Z" suffix.
        return val.parse::<Timestamp>().ok();
    }
    let tm = jiff::fmt::strtime::BrokenDownTime::parse(fmt, val).ok()?;
    if let Ok(ts) = tm.to_timestamp() {
        return Some(ts);
    }
    // No offset/zone specifier matched (e.g. a literal "Z" suffix, which
    // jiff's `%z` directive does not accept) - treat as UTC.
    let dt = tm.to_datetime().ok()?;
    jiff::tz::TimeZone::UTC.to_timestamp(dt).ok()
}

fn transform_tzaware_datetime_ns(val: &str, fmt: &str) -> Option<i64> {
    let ts = parse_tz_aware_timestamp(val, fmt)?;
    Some(datetime_to_timestamp_ns(
        jiff::tz::TimeZone::UTC.to_datetime(ts),
    ))
}

fn transform_tzaware_datetime_us(val: &str, fmt: &str) -> Option<i64> {
    let ts = parse_tz_aware_timestamp(val, fmt)?;
    Some(datetime_to_timestamp_us(
        jiff::tz::TimeZone::UTC.to_datetime(ts),
    ))
}

fn transform_tzaware_datetime_ms(val: &str, fmt: &str) -> Option<i64> {
    let ts = parse_tz_aware_timestamp(val, fmt)?;
    Some(datetime_to_timestamp_ms(
        jiff::tz::TimeZone::UTC.to_datetime(ts),
    ))
}

pub fn infer_pattern_single(val: &str) -> Option<Pattern> {
    // Dates come first, because we see datetimes as superset of dates
    infer_pattern_date_single(val)
        .or_else(|| infer_pattern_time_single(val))
        .or_else(|| infer_pattern_datetime_single(val))
}

pub fn infer_pattern_datetime_single(val: &str) -> Option<Pattern> {
    if patterns::DATETIME_D_M_Y.iter().any(|fmt| {
        NaiveDateTime::strptime(fmt, val).is_ok() || NaiveDate::strptime(fmt, val).is_ok()
    }) {
        Some(Pattern::DatetimeDMY)
    } else if patterns::DATETIME_Y_M_D.iter().any(|fmt| {
        NaiveDateTime::strptime(fmt, val).is_ok() || NaiveDate::strptime(fmt, val).is_ok()
    }) {
        Some(Pattern::DatetimeYMD)
    } else if patterns::DATETIME_Y_M_D_Z.iter().any(|fmt| {
        // "%+" isn't a directive jiff's strtime engine understands - it's
        // handled separately (see `parse_tz_aware_timestamp`) via jiff's
        // native ISO 8601 timestamp parser, which is what we must probe with
        // here too, or values only expressible via "%+" (e.g. a literal "Z"
        // offset with no colon/offset digits) would never be inferred.
        if *fmt == "%+" {
            val.parse::<Timestamp>().is_ok()
        } else {
            NaiveDateTime::strptime(fmt, val).is_ok()
        }
    }) {
        Some(Pattern::DatetimeYMDZ)
    } else {
        None
    }
}

pub fn infer_pattern_date_single(val: &str) -> Option<Pattern> {
    if patterns::DATE_D_M_Y
        .iter()
        .any(|fmt| NaiveDate::strptime(fmt, val).is_ok())
    {
        Some(Pattern::DateDMY)
    } else if patterns::DATE_Y_M_D
        .iter()
        .any(|fmt| NaiveDate::strptime(fmt, val).is_ok())
    {
        Some(Pattern::DateYMD)
    } else {
        None
    }
}

pub fn infer_pattern_time_single(val: &str) -> Option<Pattern> {
    sniff_time_fmt(val).is_some().then_some(Pattern::Time)
}

/// Return the first format string from `TIME_H_M_S` that parses `val`, or `None`.
pub fn sniff_time_fmt(val: &str) -> Option<&'static str> {
    patterns::TIME_H_M_S
        .iter()
        .copied()
        .find(|fmt| NaiveTime::strptime(fmt, val).is_ok())
}

#[cfg(feature = "dtype-datetime")]
pub fn to_datetime_with_inferred_tz(
    ca: &StringChunked,
    tu: TimeUnit,
    strict: bool,
    exact: bool,
    ambiguous: &StringChunked,
) -> PolarsResult<DatetimeChunked> {
    use super::StringMethods;

    let out = if exact {
        to_datetime(ca, tu, None, ambiguous, false)
    } else {
        ca.as_datetime_not_exact(None, tu, false, None, ambiguous, false)
    }?;

    if strict && ca.null_count() != out.null_count() {
        polars_core::utils::handle_casting_failures(
            &ca.clone().into_series(),
            &out.clone().into_series(),
        )?;
    }

    Ok(out)
}

#[cfg(feature = "dtype-datetime")]
pub fn to_datetime(
    ca: &StringChunked,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
    ambiguous: &StringChunked,
    // Ensure that the inferred time_zone matches the given time_zone.
    ensure_matching_time_zone: bool,
) -> PolarsResult<DatetimeChunked> {
    match ca.first_non_null() {
        None => {
            Ok(Int64Chunked::full_null(ca.name().clone(), ca.len()).into_datetime(tu, tz.cloned()))
        },
        Some(idx) => {
            let subset = ca.slice(idx as i64, ca.len());
            let pattern = subset
                .iter()
                .find_map(|opt_val| opt_val.and_then(infer_pattern_datetime_single))
                .ok_or_else(|| {
                    let sample = ca.get(idx);
                    if sample.is_some_and(|val| super::TRAILING_OFFSET_RE.is_match(val)) {
                        polars_err!(
                            ComputeError:
                            "could not find an appropriate format to parse datetimes, please define a format\n\n\
                            hint: this value appears to contain a UTC offset (e.g. '+01:00', '+0100', or 'Z'). \
                            Polars' format directives (%z, %:z, %::z, %:::z) each require an exact colon style, \
                            so an explicit format matching your data's offset style may be required, e.g. `%:z` \
                            for a colon-separated offset like '+01:00'.",
                        )
                    } else {
                        polars_err!(parse_fmt_idk = "date")
                    }
                })?;
            let mut infer = DatetimeInfer::<Int64Type>::try_from_with_unit(pattern, Some(tu))?;
            #[cfg(feature = "timezones")]
            if matches!(pattern, Pattern::DatetimeYMDZ) {
                polars_ensure!(
                    !ensure_matching_time_zone || tz.is_some(),
                    to_datetime_tz_mismatch
                );
            }
            coerce_string_to_datetime(&mut infer, ca, tz, ambiguous)
        },
    }
}
/// Apply a pre-built `DatetimeInfer<Int32Type>` to a `StringChunked`, returning a `DateChunked`.
#[cfg(feature = "dtype-date")]
pub fn coerce_string_to_date(
    infer: &mut DatetimeInfer<Int32Type>,
    ca: &StringChunked,
) -> PolarsResult<DateChunked> {
    infer.coerce_string(ca).date().cloned()
}

/// Apply a pre-built `DatetimeInfer<Int64Type>` to a `StringChunked`, applying tz handling,
/// returning a `DatetimeChunked`. Mirrors the post-`coerce_string` logic in `to_datetime`.
#[cfg(feature = "dtype-datetime")]
pub fn coerce_string_to_datetime(
    infer: &mut DatetimeInfer<Int64Type>,
    ca: &StringChunked,
    tz: Option<&TimeZone>,
    ambiguous: &StringChunked,
) -> PolarsResult<DatetimeChunked> {
    let DataType::Datetime(tu, _) = &infer.logical_type else {
        unreachable!()
    };
    let tu = *tu;
    match infer.pattern {
        #[cfg(feature = "timezones")]
        Pattern::DatetimeYMDZ => infer.coerce_string(ca).datetime().map(|ca| {
            let mut ca = ca.clone();
            ca.set_time_unit_and_time_zone(tu, tz.cloned().unwrap_or(TimeZone::UTC))?;
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
                    ambiguous,
                    NonExistent::Raise,
                ),
                _ => Ok(ca),
            }
        })?,
    }
}

#[cfg(feature = "dtype-date")]
pub(crate) fn to_date(ca: &StringChunked) -> PolarsResult<DateChunked> {
    match ca.first_non_null() {
        None => Ok(Int32Chunked::full_null(ca.name().clone(), ca.len()).into_date()),
        Some(idx) => {
            let subset = ca.slice(idx as i64, ca.len());
            let pattern = subset
                .iter()
                .find_map(|opt_val| opt_val.and_then(infer_pattern_date_single))
                .ok_or_else(|| polars_err!(parse_fmt_idk = "date"))?;
            let mut infer = DatetimeInfer::<Int32Type>::try_from_with_unit(pattern, None).unwrap();
            coerce_string_to_date(&mut infer, ca)
        },
    }
}
