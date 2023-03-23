#[cfg(feature = "timezones")]
use chrono_tz::Tz;
#[cfg(feature = "timezones")]
use polars_core::utils::arrow::temporal_conversions::parse_offset;
use polars_core::utils::arrow::temporal_conversions::SECONDS_IN_DAY;
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
    Round(String, String),
    #[cfg(feature = "timezones")]
    CastTimezone(Option<TimeZone>),
    #[cfg(feature = "timezones")]
    TzLocalize(TimeZone),
    DateRange {
        name: String,
        every: Duration,
        closed: ClosedWindow,
        tz: Option<TimeZone>,
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
            Round(..) => "round",
            #[cfg(feature = "timezones")]
            CastTimezone(_) => "replace_timezone",
            #[cfg(feature = "timezones")]
            TzLocalize(_) => "tz_localize",
            DateRange { .. } => return write!(f, "date_range"),
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
            .replace_time_zone(None)?
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
        DataType::Datetime(_, Some(_)) => s
            .datetime()
            .unwrap()
            .replace_time_zone(None)?
            .cast(&DataType::Date),
        DataType::Datetime(_, _) => s.datetime().unwrap().cast(&DataType::Date),
        DataType::Date => Ok(s.clone()),
        dtype => polars_bail!(ComputeError: "expected Datetime or Date, got {}", dtype),
    }
}
pub(super) fn datetime(s: &Series) -> PolarsResult<Series> {
    match s.dtype() {
        #[cfg(feature = "timezones")]
        DataType::Datetime(tu, Some(_)) => s
            .datetime()
            .unwrap()
            .replace_time_zone(None)?
            .cast(&DataType::Datetime(*tu, None)),
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
    Ok(match s.dtype() {
        DataType::Datetime(_, tz) => match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => match tz.parse::<Tz>() {
                Ok(tz) => s
                    .datetime()
                    .unwrap()
                    .truncate(every, offset, Some(&tz))?
                    .into_series(),
                Err(_) => match parse_offset(tz) {
                    Ok(tz) => s
                        .datetime()
                        .unwrap()
                        .truncate(every, offset, Some(&tz))?
                        .into_series(),
                    Err(_) => unreachable!(),
                },
            },
            _ => s
                .datetime()
                .unwrap()
                .truncate(every, offset, NO_TIMEZONE)?
                .into_series(),
        },
        DataType::Date => s
            .date()
            .unwrap()
            .truncate(every, offset, NO_TIMEZONE)?
            .into_series(),
        dt => polars_bail!(opq = round, got = dt, expected = "date/datetime"),
    })
}

pub(super) fn round(s: &Series, every: &str, offset: &str) -> PolarsResult<Series> {
    let every = Duration::parse(every);
    let offset = Duration::parse(offset);
    Ok(match s.dtype() {
        DataType::Datetime(_, tz) => match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => match tz.parse::<Tz>() {
                Ok(tz) => s
                    .datetime()
                    .unwrap()
                    .round(every, offset, Some(&tz))?
                    .into_series(),
                Err(_) => match parse_offset(tz) {
                    Ok(tz) => s
                        .datetime()
                        .unwrap()
                        .round(every, offset, Some(&tz))?
                        .into_series(),
                    Err(_) => unreachable!(),
                },
            },
            _ => s
                .datetime()
                .unwrap()
                .round(every, offset, NO_TIMEZONE)?
                .into_series(),
        },
        DataType::Date => s
            .date()
            .unwrap()
            .round(every, offset, NO_TIMEZONE)?
            .into_series(),
        dt => polars_bail!(opq = round, got = dt, expected = "date/datetime"),
    })
}

#[cfg(feature = "timezones")]
pub(super) fn replace_timezone(s: &Series, time_zone: Option<&str>) -> PolarsResult<Series> {
    let ca = s.datetime()?;
    ca.replace_time_zone(time_zone).map(|ca| ca.into_series())
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
    Ok(ca.replace_time_zone(Some(tz))?.into_series())
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

    polars_ensure!(
        start.len() == stop.len(),
        ComputeError: "'start' and 'stop' should have the same length",
    );
    const TO_MS: i64 = SECONDS_IN_DAY * 1000;

    if start.len() == 1 && stop.len() == 1 {
        match start.dtype() {
            DataType::Date => {
                let start = start.to_physical_repr();
                let stop = stop.to_physical_repr();
                // to milliseconds
                let start = start.get(0).unwrap().extract::<i64>().unwrap() * TO_MS;
                let stop = stop.get(0).unwrap().extract::<i64>().unwrap() * TO_MS;

                date_range_impl(
                    name,
                    start,
                    stop,
                    every,
                    closed,
                    TimeUnit::Milliseconds,
                    tz.as_ref(),
                )?
                .cast(&DataType::Date)
            }
            DataType::Datetime(tu, _) => {
                let start = start.to_physical_repr();
                let stop = stop.to_physical_repr();
                let start = start.get(0).unwrap().extract::<i64>().unwrap();
                let stop = stop.get(0).unwrap().extract::<i64>().unwrap();

                Ok(
                    date_range_impl(name, start, stop, every, closed, *tu, tz.as_ref())?
                        .into_series(),
                )
            }
            _ => unimplemented!(),
        }
    } else {
        let dtype = start.dtype();

        let mut start = start.to_physical_repr().cast(&DataType::Int64)?;
        let mut stop = stop.to_physical_repr().cast(&DataType::Int64)?;

        let (tu, tz) = match dtype {
            DataType::Date => {
                start = &start * TO_MS;
                stop = &stop * TO_MS;
                (TimeUnit::Milliseconds, None)
            }
            DataType::Datetime(tu, tz) => (*tu, tz.as_ref()),
            _ => unimplemented!(),
        };

        let start = start.i64().unwrap();
        let stop = stop.i64().unwrap();

        let list = match dtype {
            DataType::Date => {
                let mut builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(
                    name,
                    start.len(),
                    start.len() * 5,
                    DataType::Int32,
                );
                for (start, stop) in start.into_iter().zip(stop.into_iter()) {
                    match (start, stop) {
                        (Some(start), Some(stop)) => {
                            let date_range =
                                date_range_impl("", start, stop, every, closed, tu, tz)?;
                            let date_range = date_range.cast(&DataType::Date).unwrap();
                            let date_range = date_range.to_physical_repr();
                            let date_range = date_range.i32().unwrap();
                            builder.append_slice(date_range.cont_slice().unwrap())
                        }
                        _ => builder.append_null(),
                    }
                }
                builder.finish().into_series()
            }
            DataType::Datetime(_, _) => {
                let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
                    name,
                    start.len(),
                    start.len() * 5,
                    DataType::Int64,
                );

                for (start, stop) in start.into_iter().zip(stop.into_iter()) {
                    match (start, stop) {
                        (Some(start), Some(stop)) => {
                            let date_range =
                                date_range_impl("", start, stop, every, closed, tu, tz)?;
                            builder.append_slice(date_range.cont_slice().unwrap())
                        }
                        _ => builder.append_null(),
                    }
                }
                builder.finish().into_series()
            }
            _ => unimplemented!(),
        };

        let to_type = DataType::List(Box::new(dtype.clone()));
        list.cast(&to_type)
    }
}
