use polars_core::utils::arrow::temporal_conversions::SECONDS_IN_DAY;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum TemporalFunction {
    Year,
    IsoYear,
    Quarter,
    Month,
    Week,
    WeekDay,
    Day,
    OrdinalDay,
    Hour,
    Minute,
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
    TimeStamp(TimeUnit),
    Truncate(String, String),
    Round(String, String),
    #[cfg(feature = "timezones")]
    CastTimezone(TimeZone),
    #[cfg(feature = "timezones")]
    TzLocalize(TimeZone),
    DateRange {
        name: String,
        every: Duration,
        closed: ClosedWindow,
        tz: Option<TimeZone>,
    },
}

impl Display for TemporalFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use TemporalFunction::*;
        let s = match self {
            Year => "year",
            IsoYear => "iso_year",
            Quarter => "quarter",
            Month => "month",
            Week => "week",
            WeekDay => "weekday",
            Day => "day",
            OrdinalDay => "ordinal_day",
            Hour => "hour",
            Minute => "minute",
            Second => "second",
            Millisecond => "millisecond",
            Microsecond => "microsecond",
            Nanosecond => "nanosecond",
            TimeStamp(tu) => return write!(f, "dt.timestamp({})", tu),
            Truncate(..) => "truncate",
            Round(..) => "round",
            #[cfg(feature = "timezones")]
            CastTimezone(_) => "cast_timezone",
            #[cfg(feature = "timezones")]
            TzLocalize(_) => "tz_localize",
            DateRange { .. } => return write!(f, "date_range"),
        };
        write!(f, "dt.{}", s)
    }
}

pub(super) fn year(s: &Series) -> PolarsResult<Series> {
    s.year().map(|ca| ca.into_series())
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
    match s.dtype() {
        DataType::Datetime(_, _) => Ok(s.datetime().unwrap().truncate(every, offset).into_series()),
        DataType::Date => Ok(s.date().unwrap().truncate(every, offset).into_series()),
        dt => Err(PolarsError::ComputeError(
            format!("expected date/datetime got {:?}", dt).into(),
        )),
    }
}
pub(super) fn round(s: &Series, every: &str, offset: &str) -> PolarsResult<Series> {
    let every = Duration::parse(every);
    let offset = Duration::parse(offset);
    match s.dtype() {
        DataType::Datetime(_, _) => Ok(s.datetime().unwrap().round(every, offset).into_series()),
        DataType::Date => Ok(s.date().unwrap().round(every, offset).into_series()),
        dt => Err(PolarsError::ComputeError(
            format!("expected date/datetime got {:?}", dt).into(),
        )),
    }
}
#[cfg(feature = "timezones")]
pub(super) fn cast_timezone(s: &Series, tz: &str) -> PolarsResult<Series> {
    let ca = s.datetime()?;
    ca.cast_time_zone(tz).map(|ca| ca.into_series())
}

#[cfg(feature = "timezones")]
pub(super) fn tz_localize(s: &Series, tz: &str) -> PolarsResult<Series> {
    let ca = s.datetime()?.clone();
    match ca.time_zone() {
        Some(tz) if !tz.is_empty() => {
            Err(PolarsError::ComputeError("Cannot localize a tz-aware datetime. Consider using 'dt.with_time_zone' or 'dt.cast_time_zone'".into()))
        },
        _ => {
            Ok(ca.with_time_zone(Some(tz.into())).cast_time_zone("UTC")?.with_time_zone(Some(tz.into())).into_series())
        }
    }
}

pub(super) fn date_range_dispatch(
    s: &[Series],
    name: &str,
    every: Duration,
    closed: ClosedWindow,
    tz: Option<TimeZone>,
) -> PolarsResult<Series> {
    let start = &s[0];
    let stop = &s[1];

    match start.dtype() {
        DataType::Date => {
            let start = start.to_physical_repr();
            let stop = stop.to_physical_repr();
            // to milliseconds
            let start = start.get(0).extract::<i64>().unwrap() * SECONDS_IN_DAY * 1000;
            let stop = stop.get(0).extract::<i64>().unwrap() * SECONDS_IN_DAY * 1000;

            date_range_impl(name, start, stop, every, closed, TimeUnit::Milliseconds, tz)
                .cast(&DataType::Date)
        }
        DataType::Datetime(tu, _) => {
            let start = start.to_physical_repr();
            let stop = stop.to_physical_repr();
            let start = start.get(0).extract::<i64>().unwrap();
            let stop = stop.get(0).extract::<i64>().unwrap();

            Ok(date_range_impl(name, start, stop, every, closed, *tu, tz).into_series())
        }
        _ => todo!(),
    }
}
