#[cfg(feature = "timezones")]
use chrono_tz::Tz;
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
    Truncate(String, String),
    #[cfg(feature = "date_offset")]
    MonthStart,
    #[cfg(feature = "date_offset")]
    MonthEnd,
    Round(String, String),
    #[cfg(feature = "timezones")]
    CastTimezone(Option<TimeZone>, Option<bool>),
    #[cfg(feature = "timezones")]
    TzLocalize(TimeZone),
    DateRange {
        every: Duration,
        closed: ClosedWindow,
        tz: Option<TimeZone>,
    },
    TimeRange {
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
            Round(..) => "round",
            #[cfg(feature = "timezones")]
            CastTimezone(_, _) => "replace_timezone",
            #[cfg(feature = "timezones")]
            TzLocalize(_) => "tz_localize",
            DateRange { .. } => return write!(f, "date_range"),
            TimeRange { .. } => return write!(f, "time_range"),
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
        DataType::Datetime(_, Some(_)) => s
            .datetime()
            .unwrap()
            .replace_time_zone(None, None)?
            .cast(&DataType::Time),
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
            let mut out = s
                .datetime()
                .unwrap()
                .replace_time_zone(None, None)?
                .cast(&DataType::Date)?;
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
            let mut out = s
                .datetime()
                .unwrap()
                .replace_time_zone(None, None)?
                .cast(&DataType::Datetime(*tu, None))?;
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

pub(super) fn truncate(s: &Series, every: &str, offset: &str) -> PolarsResult<Series> {
    let every = Duration::parse(every);
    let offset = Duration::parse(offset);
    let mut out = match s.dtype() {
        DataType::Datetime(_, tz) => match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => s
                .datetime()
                .unwrap()
                .truncate(every, offset, tz.parse::<Tz>().ok().as_ref())?
                .into_series(),
            _ => s
                .datetime()
                .unwrap()
                .truncate(every, offset, None)?
                .into_series(),
        },
        DataType::Date => s
            .date()
            .unwrap()
            .truncate(every, offset, None)?
            .into_series(),
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

#[cfg(feature = "timezones")]
pub(super) fn replace_timezone(
    s: &Series,
    time_zone: Option<&str>,
    use_earliest: Option<bool>,
) -> PolarsResult<Series> {
    let ca = s.datetime()?;
    ca.replace_time_zone(time_zone, use_earliest)
        .map(|ca| ca.into_series())
}

#[cfg(feature = "timezones")]
#[deprecated(note = "use replace_time_zone")]
pub(super) fn tz_localize(s: &Series, tz: &str) -> PolarsResult<Series> {
    let ca = s.datetime()?.clone();
    polars_ensure!(
        ca.time_zone().as_ref().map_or(true, |tz| tz.is_empty()),
        ComputeError:
        "cannot localize a tz-aware datetime \
        (consider using 'dt.convert_time_zone' or 'dt.replace_time_zone')"
    );
    Ok(ca.replace_time_zone(Some(tz), None)?.into_series())
}
