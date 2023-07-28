#[cfg(feature = "timezones")]
use chrono_tz::Tz;
#[cfg(feature = "timezones")]
use polars_time::base_utc_offset as base_utc_offset_fn;
#[cfg(feature = "timezones")]
use polars_time::dst_offset as dst_offset_fn;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum TemporalFunction {
    Year,
    IsLeapYear,
    IsoYear,
    Quarter,
    Month,
    Week,
    WeekDay,
    Day,
    OrdinalDay,
    Time,
    Date,
    Datetime,
    Hour,
    Minute,
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
    TimeStamp(TimeUnit),
    Truncate(TruncateOptions),
    #[cfg(feature = "date_offset")]
    MonthStart,
    #[cfg(feature = "date_offset")]
    MonthEnd,
    #[cfg(feature = "timezones")]
    BaseUtcOffset,
    #[cfg(feature = "timezones")]
    DSTOffset,
    Round(String, String),
    #[cfg(feature = "timezones")]
    ReplaceTimeZone(Option<TimeZone>, Option<bool>),
    DateRange {
        every: Duration,
        closed: ClosedWindow,
        time_unit: Option<TimeUnit>,
        time_zone: Option<TimeZone>,
    },
    DateRanges {
        every: Duration,
        closed: ClosedWindow,
        time_unit: Option<TimeUnit>,
        time_zone: Option<TimeZone>,
    },
    TimeRange {
        every: Duration,
        closed: ClosedWindow,
    },
    TimeRanges {
        every: Duration,
        closed: ClosedWindow,
    },
    Combine(TimeUnit),
}

impl Display for TemporalFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use TemporalFunction::*;
        let s = match self {
            Year => "year",
            IsLeapYear => "is_leap_year",
            IsoYear => "iso_year",
            Quarter => "quarter",
            Month => "month",
            Week => "week",
            WeekDay => "weekday",
            Day => "day",
            OrdinalDay => "ordinal_day",
            Time => "time",
            Date => "date",
            Datetime => "datetime",
            Hour => "hour",
            Minute => "minute",
            Second => "second",
            Millisecond => "millisecond",
            Microsecond => "microsecond",
            Nanosecond => "nanosecond",
            TimeStamp(tu) => return write!(f, "dt.timestamp({tu})"),
            Truncate(..) => "truncate",
            #[cfg(feature = "date_offset")]
            MonthStart => "month_start",
            #[cfg(feature = "date_offset")]
            MonthEnd => "month_end",
            #[cfg(feature = "timezones")]
            BaseUtcOffset => "base_utc_offset",
            #[cfg(feature = "timezones")]
            DSTOffset => "dst_offset",
            Round(..) => "round",
            #[cfg(feature = "timezones")]
            ReplaceTimeZone(_, _) => "replace_time_zone",
            DateRange { .. } => return write!(f, "date_range"),
            DateRanges { .. } => return write!(f, "date_ranges"),
            TimeRange { .. } => return write!(f, "time_range"),
            TimeRanges { .. } => return write!(f, "time_ranges"),
            Combine(_) => "combine",
        };
        write!(f, "dt.{s}")
    }
}

pub(super) fn year(s: &Series) -> PolarsResult<Series> {
    s.year().map(|ca| ca.into_series())
}
pub(super) fn is_leap_year(s: &Series) -> PolarsResult<Series> {
    s.is_leap_year().map(|ca| ca.into_series())
}
pub(super) fn iso_year(s: &Series) -> PolarsResult<Series> {
    s.iso_year().map(|ca| ca.into_series())
}
pub(super) fn month(s: &Series) -> PolarsResult<Series> {
    s.month().map(|ca| ca.into_series())
}
pub(super) fn quarter(s: &Series) -> PolarsResult<Series> {
    s.quarter().map(|ca| ca.into_series())
}
pub(super) fn week(s: &Series) -> PolarsResult<Series> {
    s.week().map(|ca| ca.into_series())
}
pub(super) fn weekday(s: &Series) -> PolarsResult<Series> {
    s.weekday().map(|ca| ca.into_series())
}
pub(super) fn day(s: &Series) -> PolarsResult<Series> {
    s.day().map(|ca| ca.into_series())
}
pub(super) fn ordinal_day(s: &Series) -> PolarsResult<Series> {
    s.ordinal_day().map(|ca| ca.into_series())
}
pub(super) fn time(s: &Series) -> PolarsResult<Series> {
    match s.dtype() {
        #[cfg(feature = "timezones")]
        DataType::Datetime(_, Some(_)) => {
            polars_ops::prelude::replace_time_zone(s.datetime().unwrap(), None, None)?
                .cast(&DataType::Time)
        }
        DataType::Datetime(_, _) => s.datetime().unwrap().cast(&DataType::Time),
        DataType::Date => s.datetime().unwrap().cast(&DataType::Time),
        DataType::Time => Ok(s.clone()),
        dtype => polars_bail!(ComputeError: "expected Datetime, Date, or Time, got {}", dtype),
    }
}
pub(super) fn date(s: &Series) -> PolarsResult<Series> {
    match s.dtype() {
        #[cfg(feature = "timezones")]
        DataType::Datetime(_, Some(tz)) => {
            let mut out = {
                polars_ops::chunked_array::replace_time_zone(s.datetime().unwrap(), None, None)?
                    .cast(&DataType::Date)?
            };
            if tz != "UTC" {
                // DST transitions may not preserve sortedness.
                out.set_sorted_flag(IsSorted::Not);
            }
            Ok(out)
        }
        DataType::Datetime(_, _) => s.datetime().unwrap().cast(&DataType::Date),
        DataType::Date => Ok(s.clone()),
        dtype => polars_bail!(ComputeError: "expected Datetime or Date, got {}", dtype),
    }
}
pub(super) fn datetime(s: &Series) -> PolarsResult<Series> {
    match s.dtype() {
        #[cfg(feature = "timezones")]
        DataType::Datetime(tu, Some(tz)) => {
            let mut out = {
                polars_ops::chunked_array::replace_time_zone(s.datetime().unwrap(), None, None)?
                    .cast(&DataType::Datetime(*tu, None))?
            };
            if tz != "UTC" {
                // DST transitions may not preserve sortedness.
                out.set_sorted_flag(IsSorted::Not);
            }
            Ok(out)
        }
        DataType::Datetime(tu, _) => s.datetime().unwrap().cast(&DataType::Datetime(*tu, None)),
        dtype => polars_bail!(ComputeError: "expected Datetime, got {}", dtype),
    }
}
pub(super) fn hour(s: &Series) -> PolarsResult<Series> {
    s.hour().map(|ca| ca.into_series())
}
pub(super) fn minute(s: &Series) -> PolarsResult<Series> {
    s.minute().map(|ca| ca.into_series())
}
pub(super) fn second(s: &Series) -> PolarsResult<Series> {
    s.second().map(|ca| ca.into_series())
}
pub(super) fn millisecond(s: &Series) -> PolarsResult<Series> {
    s.nanosecond().map(|ca| (ca / 1_000_000).into_series())
}
pub(super) fn microsecond(s: &Series) -> PolarsResult<Series> {
    s.nanosecond().map(|ca| (ca / 1_000).into_series())
}
pub(super) fn nanosecond(s: &Series) -> PolarsResult<Series> {
    s.nanosecond().map(|ca| ca.into_series())
}
pub(super) fn timestamp(s: &Series, tu: TimeUnit) -> PolarsResult<Series> {
    s.timestamp(tu).map(|ca| ca.into_series())
}

pub(super) fn truncate(s: &Series, options: &TruncateOptions) -> PolarsResult<Series> {
    let mut out = match s.dtype() {
        DataType::Datetime(_, tz) => match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => s
                .datetime()
                .unwrap()
                .truncate(options, tz.parse::<Tz>().ok().as_ref())?
                .into_series(),
            _ => s.datetime().unwrap().truncate(options, None)?.into_series(),
        },
        DataType::Date => s.date().unwrap().truncate(options, None)?.into_series(),
        dt => polars_bail!(opq = round, got = dt, expected = "date/datetime"),
    };
    out.set_sorted_flag(s.is_sorted_flag());
    Ok(out)
}

#[cfg(feature = "date_offset")]
pub(super) fn month_start(s: &Series) -> PolarsResult<Series> {
    Ok(match s.dtype() {
        DataType::Datetime(_, tz) => match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => s
                .datetime()
                .unwrap()
                .month_start(tz.parse::<Tz>().ok().as_ref())?
                .into_series(),
            _ => s.datetime().unwrap().month_start(None)?.into_series(),
        },
        DataType::Date => s.date().unwrap().month_start(None)?.into_series(),
        dt => polars_bail!(opq = month_start, got = dt, expected = "date/datetime"),
    })
}

#[cfg(feature = "date_offset")]
pub(super) fn month_end(s: &Series) -> PolarsResult<Series> {
    Ok(match s.dtype() {
        DataType::Datetime(_, tz) => match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => s
                .datetime()
                .unwrap()
                .month_end(tz.parse::<Tz>().ok().as_ref())?
                .into_series(),
            _ => s.datetime().unwrap().month_end(None)?.into_series(),
        },
        DataType::Date => s.date().unwrap().month_end(None)?.into_series(),
        dt => polars_bail!(opq = month_end, got = dt, expected = "date/datetime"),
    })
}

#[cfg(feature = "timezones")]
pub(super) fn base_utc_offset(s: &Series) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Datetime(time_unit, Some(tz)) => {
            let tz = tz
                .parse::<Tz>()
                .expect("Time zone has already been validated");
            Ok(base_utc_offset_fn(s.datetime().unwrap(), time_unit, &tz).into_series())
        }
        dt => polars_bail!(
            opq = base_utc_offset,
            got = dt,
            expected = "time-zone-aware datetime"
        ),
    }
}
#[cfg(feature = "timezones")]
pub(super) fn dst_offset(s: &Series) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Datetime(time_unit, Some(tz)) => {
            let tz = tz
                .parse::<Tz>()
                .expect("Time zone has already been validated");
            Ok(dst_offset_fn(s.datetime().unwrap(), time_unit, &tz).into_series())
        }
        dt => polars_bail!(
            opq = dst_offset,
            got = dt,
            expected = "time-zone-aware datetime"
        ),
    }
}

pub(super) fn round(s: &Series, every: &str, offset: &str) -> PolarsResult<Series> {
    let every = Duration::parse(every);
    let offset = Duration::parse(offset);
    Ok(match s.dtype() {
        DataType::Datetime(_, tz) => match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => s
                .datetime()
                .unwrap()
                .round(every, offset, tz.parse::<Tz>().ok().as_ref())?
                .into_series(),
            _ => s
                .datetime()
                .unwrap()
                .round(every, offset, None)?
                .into_series(),
        },
        DataType::Date => s.date().unwrap().round(every, offset, None)?.into_series(),
        dt => polars_bail!(opq = round, got = dt, expected = "date/datetime"),
    })
}
