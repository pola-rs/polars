#[cfg(feature = "timezones")]
use chrono_tz::Tz;
#[cfg(feature = "timezones")]
use polars_core::chunked_array::temporal::validate_time_zone;
#[cfg(feature = "timezones")]
use polars_time::base_utc_offset as base_utc_offset_fn;
#[cfg(feature = "timezones")]
use polars_time::dst_offset as dst_offset_fn;
#[cfg(feature = "offset_by")]
use polars_time::impl_offset_by;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum TemporalFunction {
    Millennium,
    Century,
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
    Duration(TimeUnit),
    Hour,
    Minute,
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
    TotalDays,
    TotalHours,
    TotalMinutes,
    TotalSeconds,
    TotalMilliseconds,
    TotalMicroseconds,
    TotalNanoseconds,
    ToString(String),
    CastTimeUnit(TimeUnit),
    WithTimeUnit(TimeUnit),
    #[cfg(feature = "timezones")]
    ConvertTimeZone(TimeZone),
    TimeStamp(TimeUnit),
    Truncate,
    #[cfg(feature = "offset_by")]
    OffsetBy,
    #[cfg(feature = "month_start")]
    MonthStart,
    #[cfg(feature = "month_end")]
    MonthEnd,
    #[cfg(feature = "timezones")]
    BaseUtcOffset,
    #[cfg(feature = "timezones")]
    DSTOffset,
    Round,
    #[cfg(feature = "timezones")]
    ReplaceTimeZone(Option<TimeZone>, NonExistent),
    Combine(TimeUnit),
    DatetimeFunction {
        time_unit: TimeUnit,
        time_zone: Option<TimeZone>,
    },
}

impl TemporalFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use TemporalFunction::*;
        match self {
            Millennium | Century => mapper.with_dtype(DataType::Int8),
            Year | IsoYear => mapper.with_dtype(DataType::Int32),
            OrdinalDay => mapper.with_dtype(DataType::Int16),
            Month | Quarter | Week | WeekDay | Day | Hour | Minute | Second => {
                mapper.with_dtype(DataType::Int8)
            },
            Millisecond | Microsecond | Nanosecond => mapper.with_dtype(DataType::Int32),
            TotalDays | TotalHours | TotalMinutes | TotalSeconds | TotalMilliseconds
            | TotalMicroseconds | TotalNanoseconds => mapper.with_dtype(DataType::Int64),
            ToString(_) => mapper.with_dtype(DataType::String),
            WithTimeUnit(_) => mapper.with_same_dtype(),
            CastTimeUnit(tu) => mapper.try_map_dtype(|dt| match dt {
                DataType::Duration(_) => Ok(DataType::Duration(*tu)),
                DataType::Datetime(_, tz) => Ok(DataType::Datetime(*tu, tz.clone())),
                dtype => polars_bail!(ComputeError: "expected duration or datetime, got {}", dtype),
            }),
            #[cfg(feature = "timezones")]
            ConvertTimeZone(tz) => mapper.try_map_dtype(|dt| match dt {
                DataType::Datetime(tu, _) => Ok(DataType::Datetime(*tu, Some(tz.clone()))),
                dtype => polars_bail!(ComputeError: "expected Datetime, got {}", dtype),
            }),
            TimeStamp(_) => mapper.with_dtype(DataType::Int64),
            IsLeapYear => mapper.with_dtype(DataType::Boolean),
            Time => mapper.with_dtype(DataType::Time),
            Duration(tu) => mapper.with_dtype(DataType::Duration(*tu)),
            Date => mapper.with_dtype(DataType::Date),
            Datetime => mapper.try_map_dtype(|dt| match dt {
                DataType::Datetime(tu, _) => Ok(DataType::Datetime(*tu, None)),
                dtype => polars_bail!(ComputeError: "expected Datetime, got {}", dtype),
            }),
            Truncate => mapper.with_same_dtype(),
            #[cfg(feature = "offset_by")]
            OffsetBy => mapper.with_same_dtype(),
            #[cfg(feature = "month_start")]
            MonthStart => mapper.with_same_dtype(),
            #[cfg(feature = "month_end")]
            MonthEnd => mapper.with_same_dtype(),
            #[cfg(feature = "timezones")]
            BaseUtcOffset => mapper.with_dtype(DataType::Duration(TimeUnit::Milliseconds)),
            #[cfg(feature = "timezones")]
            DSTOffset => mapper.with_dtype(DataType::Duration(TimeUnit::Milliseconds)),
            Round => mapper.with_same_dtype(),
            #[cfg(feature = "timezones")]
            ReplaceTimeZone(tz, _non_existent) => mapper.map_datetime_dtype_timezone(tz.as_ref()),
            DatetimeFunction {
                time_unit,
                time_zone,
            } => Ok(Field::new(
                "datetime",
                DataType::Datetime(*time_unit, time_zone.clone()),
            )),
            Combine(tu) => mapper.try_map_dtype(|dt| match dt {
                DataType::Datetime(_, tz) => Ok(DataType::Datetime(*tu, tz.clone())),
                DataType::Date => Ok(DataType::Datetime(*tu, None)),
                dtype => {
                    polars_bail!(ComputeError: "expected Date or Datetime, got {}", dtype)
                },
            }),
        }
    }
}

impl Display for TemporalFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use TemporalFunction::*;
        let s = match self {
            Millennium => "millennium",
            Century => "century",
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
            Duration(_) => "duration",
            Hour => "hour",
            Minute => "minute",
            Second => "second",
            Millisecond => "millisecond",
            Microsecond => "microsecond",
            Nanosecond => "nanosecond",
            TotalDays => "total_days",
            TotalHours => "total_hours",
            TotalMinutes => "total_minutes",
            TotalSeconds => "total_seconds",
            TotalMilliseconds => "total_milliseconds",
            TotalMicroseconds => "total_microseconds",
            TotalNanoseconds => "total_nanoseconds",
            ToString(_) => "to_string",
            #[cfg(feature = "timezones")]
            ConvertTimeZone(_) => "convert_time_zone",
            CastTimeUnit(_) => "cast_time_unit",
            WithTimeUnit(_) => "with_time_unit",
            TimeStamp(tu) => return write!(f, "dt.timestamp({tu})"),
            Truncate => "truncate",
            #[cfg(feature = "offset_by")]
            OffsetBy => "offset_by",
            #[cfg(feature = "month_start")]
            MonthStart => "month_start",
            #[cfg(feature = "month_end")]
            MonthEnd => "month_end",
            #[cfg(feature = "timezones")]
            BaseUtcOffset => "base_utc_offset",
            #[cfg(feature = "timezones")]
            DSTOffset => "dst_offset",
            Round => "round",
            #[cfg(feature = "timezones")]
            ReplaceTimeZone(_, _) => "replace_time_zone",
            DatetimeFunction { .. } => return write!(f, "dt.datetime"),
            Combine(_) => "combine",
        };
        write!(f, "dt.{s}")
    }
}

pub(super) fn millennium(s: &Series) -> PolarsResult<Series> {
    s.millennium().map(|ca| ca.into_series())
}
pub(super) fn century(s: &Series) -> PolarsResult<Series> {
    s.century().map(|ca| ca.into_series())
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
        DataType::Datetime(_, Some(_)) => polars_ops::prelude::replace_time_zone(
            s.datetime().unwrap(),
            None,
            &StringChunked::from_iter(std::iter::once("raise")),
            NonExistent::Raise,
        )?
        .cast(&DataType::Time),
        DataType::Datetime(_, _) => s.datetime().unwrap().cast(&DataType::Time),
        DataType::Time => Ok(s.clone()),
        dtype => polars_bail!(ComputeError: "expected Datetime or Time, got {}", dtype),
    }
}
pub(super) fn date(s: &Series) -> PolarsResult<Series> {
    match s.dtype() {
        #[cfg(feature = "timezones")]
        DataType::Datetime(_, Some(tz)) => {
            let mut out = {
                polars_ops::chunked_array::replace_time_zone(
                    s.datetime().unwrap(),
                    None,
                    &StringChunked::from_iter(std::iter::once("raise")),
                    NonExistent::Raise,
                )?
                .cast(&DataType::Date)?
            };
            if tz != "UTC" {
                // DST transitions may not preserve sortedness.
                out.set_sorted_flag(IsSorted::Not);
            }
            Ok(out)
        },
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
                polars_ops::chunked_array::replace_time_zone(
                    s.datetime().unwrap(),
                    None,
                    &StringChunked::from_iter(std::iter::once("raise")),
                    NonExistent::Raise,
                )?
                .cast(&DataType::Datetime(*tu, None))?
            };
            if tz != "UTC" {
                // DST transitions may not preserve sortedness.
                out.set_sorted_flag(IsSorted::Not);
            }
            Ok(out)
        },
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
    s.nanosecond()
        .map(|ca| (ca.wrapping_trunc_div_scalar(1_000_000)).into_series())
}
pub(super) fn microsecond(s: &Series) -> PolarsResult<Series> {
    s.nanosecond()
        .map(|ca| (ca.wrapping_trunc_div_scalar(1_000)).into_series())
}
pub(super) fn nanosecond(s: &Series) -> PolarsResult<Series> {
    s.nanosecond().map(|ca| ca.into_series())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_days(s: &Series) -> PolarsResult<Series> {
    s.duration().map(|ca| ca.days().into_series())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_hours(s: &Series) -> PolarsResult<Series> {
    s.duration().map(|ca| ca.hours().into_series())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_minutes(s: &Series) -> PolarsResult<Series> {
    s.duration().map(|ca| ca.minutes().into_series())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_seconds(s: &Series) -> PolarsResult<Series> {
    s.duration().map(|ca| ca.seconds().into_series())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_milliseconds(s: &Series) -> PolarsResult<Series> {
    s.duration().map(|ca| ca.milliseconds().into_series())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_microseconds(s: &Series) -> PolarsResult<Series> {
    s.duration().map(|ca| ca.microseconds().into_series())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_nanoseconds(s: &Series) -> PolarsResult<Series> {
    s.duration().map(|ca| ca.nanoseconds().into_series())
}
pub(super) fn timestamp(s: &Series, tu: TimeUnit) -> PolarsResult<Series> {
    s.timestamp(tu).map(|ca| ca.into_series())
}
pub(super) fn to_string(s: &Series, format: &str) -> PolarsResult<Series> {
    TemporalMethods::to_string(s, format)
}

#[cfg(feature = "timezones")]
pub(super) fn convert_time_zone(s: &Series, time_zone: &TimeZone) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Datetime(_, _) => {
            let mut ca = s.datetime()?.clone();
            validate_time_zone(time_zone)?;
            ca.set_time_zone(time_zone.clone())?;
            Ok(ca.into_series())
        },
        dtype => polars_bail!(ComputeError: "expected Datetime, got {}", dtype),
    }
}
pub(super) fn with_time_unit(s: &Series, tu: TimeUnit) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Datetime(_, _) => {
            let mut ca = s.datetime()?.clone();
            ca.set_time_unit(tu);
            Ok(ca.into_series())
        },
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(_) => {
            let mut ca = s.duration()?.clone();
            ca.set_time_unit(tu);
            Ok(ca.into_series())
        },
        dt => polars_bail!(ComputeError: "dtype `{}` has no time unit", dt),
    }
}
pub(super) fn cast_time_unit(s: &Series, tu: TimeUnit) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Datetime(_, _) => {
            let ca = s.datetime()?;
            Ok(ca.cast_time_unit(tu).into_series())
        },
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(_) => {
            let ca = s.duration()?;
            Ok(ca.cast_time_unit(tu).into_series())
        },
        dt => polars_bail!(ComputeError: "dtype `{}` has no time unit", dt),
    }
}

pub(super) fn truncate(s: &[Series]) -> PolarsResult<Series> {
    let time_series = &s[0];
    let every = s[1].str()?;

    let mut out = match time_series.dtype() {
        DataType::Datetime(_, tz) => match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => time_series
                .datetime()?
                .truncate(tz.parse::<Tz>().ok().as_ref(), every)?
                .into_series(),
            _ => time_series.datetime()?.truncate(None, every)?.into_series(),
        },
        DataType::Date => time_series.date()?.truncate(None, every)?.into_series(),
        dt => polars_bail!(opq = round, got = dt, expected = "date/datetime"),
    };
    out.set_sorted_flag(time_series.is_sorted_flag());
    Ok(out)
}

#[cfg(feature = "offset_by")]
pub(super) fn offset_by(s: &[Series]) -> PolarsResult<Series> {
    impl_offset_by(&s[0], &s[1])
}

#[cfg(feature = "month_start")]
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

#[cfg(feature = "month_end")]
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
        },
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
        },
        dt => polars_bail!(
            opq = dst_offset,
            got = dt,
            expected = "time-zone-aware datetime"
        ),
    }
}

pub(super) fn round(s: &[Series]) -> PolarsResult<Series> {
    let time_series = &s[0];
    let every = s[1].str()?;

    Ok(match time_series.dtype() {
        DataType::Datetime(_, tz) => match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => time_series
                .datetime()
                .unwrap()
                .round(every, tz.parse::<Tz>().ok().as_ref())?
                .into_series(),
            _ => time_series
                .datetime()
                .unwrap()
                .round(every, None)?
                .into_series(),
        },
        DataType::Date => time_series
            .date()
            .unwrap()
            .round(every, None)?
            .into_series(),
        dt => polars_bail!(opq = round, got = dt, expected = "date/datetime"),
    })
}
