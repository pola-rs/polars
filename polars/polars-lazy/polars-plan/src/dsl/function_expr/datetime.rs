use polars_core::export::arrow::temporal_conversions::NANOSECONDS;
use polars_core::export::chrono::NaiveDate;
use polars_core::utils::arrow::temporal_conversions::SECONDS_IN_DAY;
use polars_core::utils::CustomIterTools;
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
    /// Convert to Datetime
    ToDatetime,
    /// Convert to Duration
    ToDuration,
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
            TimeStamp(tu) => return write!(f, "dt.timestamp({tu})"),
            Truncate(..) => "truncate",
            Round(..) => "round",
            #[cfg(feature = "timezones")]
            CastTimezone(_) => "cast_timezone",
            #[cfg(feature = "timezones")]
            TzLocalize(_) => "tz_localize",
            DateRange { .. } => return write!(f, "date_range"),
            ToDatetime => return write!(f, "datetime"),
            ToDuration => return write!(f, "duration"),
        };
        write!(f, "dt.{s}")
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
            format!("expected date/datetime got {dt:?}").into(),
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
            format!("expected date/datetime got {dt:?}").into(),
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
        Some(tz) if tz == "UTC" => {
           Ok(ca.with_time_zone(Some("UTC".into())).into_series())
        }
        Some(tz) if !tz.is_empty() => {
            Err(PolarsError::ComputeError("Cannot localize a tz-aware datetime. Consider using 'dt.with_time_zone' or 'dt.cast_time_zone'".into()))
        },
        _ => {
            Ok(ca.with_time_zone(Some("UTC".to_string())).cast_time_zone(tz)?.into_series())
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

    if start.len() != stop.len() {
        return Err(PolarsError::ComputeError(
            "'start' and 'stop' should have the same length.".into(),
        ));
    }
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
                )
                .cast(&DataType::Date)
            }
            DataType::Datetime(tu, _) => {
                let start = start.to_physical_repr();
                let stop = stop.to_physical_repr();
                let start = start.get(0).unwrap().extract::<i64>().unwrap();
                let stop = stop.get(0).unwrap().extract::<i64>().unwrap();

                Ok(
                    date_range_impl(name, start, stop, every, closed, *tu, tz.as_ref())
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
                                date_range_impl("", start, stop, every, closed, tu, tz);
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
                                date_range_impl("", start, stop, every, closed, tu, tz);
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

pub(super) fn to_datetime(s: &mut [Series]) -> PolarsResult<Series> {
    assert_eq!(s.len(), 7);
    let max_len = s.iter().map(|s| s.len()).max().unwrap();
    let mut year = s[0].cast(&DataType::Int32)?;
    if year.len() < max_len {
        year = year.new_from_index(0, max_len)
    }
    let year = year.i32()?;
    let mut month = s[1].cast(&DataType::UInt32)?;
    if month.len() < max_len {
        month = month.new_from_index(0, max_len);
    }
    let month = month.u32()?;
    let mut day = s[2].cast(&DataType::UInt32)?;
    if day.len() < max_len {
        day = day.new_from_index(0, max_len);
    }
    let day = day.u32()?;
    let mut hour = s[3].cast(&DataType::UInt32)?;
    if hour.len() < max_len {
        hour = hour.new_from_index(0, max_len);
    }
    let hour = hour.u32()?;

    let mut minute = s[4].cast(&DataType::UInt32)?;
    if minute.len() < max_len {
        minute = minute.new_from_index(0, max_len);
    }
    let minute = minute.u32()?;

    let mut second = s[5].cast(&DataType::UInt32)?;
    if second.len() < max_len {
        second = second.new_from_index(0, max_len);
    }
    let second = second.u32()?;

    let mut microsecond = s[6].cast(&DataType::UInt32)?;
    if microsecond.len() < max_len {
        microsecond = microsecond.new_from_index(0, max_len);
    }
    let microsecond = microsecond.u32()?;

    let ca: Int64Chunked = year
        .into_iter()
        .zip(month.into_iter())
        .zip(day.into_iter())
        .zip(hour.into_iter())
        .zip(minute.into_iter())
        .zip(second.into_iter())
        .zip(microsecond.into_iter())
        .map(|((((((y, m), d), h), mnt), s), us)| {
            if let (Some(y), Some(m), Some(d), Some(h), Some(mnt), Some(s), Some(us)) =
                (y, m, d, h, mnt, s, us)
            {
                NaiveDate::from_ymd_opt(y, m, d)
                    .and_then(|nd| nd.and_hms_micro_opt(h, mnt, s, us))
                    .map(|ndt| ndt.timestamp_micros())
            } else {
                None
            }
        })
        .collect_trusted();

    Ok(ca.into_datetime(TimeUnit::Microseconds, None).into_series())
}

pub(super) fn to_duration(s: &mut [Series]) -> PolarsResult<Series> {
    assert_eq!(s.len(), 8);
    let days = s[0].cast(&DataType::Int64).unwrap();
    let seconds = s[1].cast(&DataType::Int64).unwrap();
    let mut nanoseconds = s[2].cast(&DataType::Int64).unwrap();
    let microseconds = s[3].cast(&DataType::Int64).unwrap();
    let milliseconds = s[4].cast(&DataType::Int64).unwrap();
    let minutes = s[5].cast(&DataType::Int64).unwrap();
    let hours = s[6].cast(&DataType::Int64).unwrap();
    let weeks = s[7].cast(&DataType::Int64).unwrap();

    let max_len = s.iter().map(|s| s.len()).max().unwrap();

    let condition = |s: &Series| {
        // check if not literal 0 || full column
        (s.len() != max_len && s.get(0).unwrap() != AnyValue::Int64(0)) || s.len() == max_len
    };

    if nanoseconds.len() != max_len {
        nanoseconds = nanoseconds.new_from_index(0, max_len);
    }
    if condition(&microseconds) {
        nanoseconds = nanoseconds + (microseconds * 1_000);
    }
    if condition(&milliseconds) {
        nanoseconds = nanoseconds + (milliseconds * 1_000_000);
    }
    if condition(&seconds) {
        nanoseconds = nanoseconds + (seconds * NANOSECONDS);
    }
    if condition(&days) {
        nanoseconds = nanoseconds + (days * NANOSECONDS * SECONDS_IN_DAY);
    }
    if condition(&minutes) {
        nanoseconds = nanoseconds + minutes * NANOSECONDS * 60;
    }
    if condition(&hours) {
        nanoseconds = nanoseconds + hours * NANOSECONDS * 60 * 60;
    }
    if condition(&weeks) {
        nanoseconds = nanoseconds + weeks * NANOSECONDS * SECONDS_IN_DAY * 7;
    }

    nanoseconds.cast(&DataType::Duration(TimeUnit::Nanoseconds))
}
