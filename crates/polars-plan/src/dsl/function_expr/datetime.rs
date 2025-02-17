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
#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
use polars_time::replace::{replace_date, replace_datetime};
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
    Replace,
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
            Replace => mapper.with_same_dtype(),
            #[cfg(feature = "timezones")]
            ReplaceTimeZone(tz, _non_existent) => mapper.map_datetime_dtype_timezone(tz.as_ref()),
            DatetimeFunction {
                time_unit,
                time_zone,
            } => Ok(Field::new(
                PlSmallStr::from_static("datetime"),
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
            Replace => "replace",
            #[cfg(feature = "timezones")]
            ReplaceTimeZone(_, _) => "replace_time_zone",
            DatetimeFunction { .. } => return write!(f, "dt.datetime"),
            Combine(_) => "combine",
        };
        write!(f, "dt.{s}")
    }
}

pub(super) fn millennium(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .millennium()
        .map(|ca| ca.into_column())
}
pub(super) fn century(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .century()
        .map(|ca| ca.into_column())
}
pub(super) fn year(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series().year().map(|ca| ca.into_column())
}
pub(super) fn is_leap_year(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .is_leap_year()
        .map(|ca| ca.into_column())
}
pub(super) fn iso_year(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .iso_year()
        .map(|ca| ca.into_column())
}
pub(super) fn month(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .month()
        .map(|ca| ca.into_column())
}
pub(super) fn quarter(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .quarter()
        .map(|ca| ca.into_column())
}
pub(super) fn week(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series().week().map(|ca| ca.into_column())
}
pub(super) fn weekday(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .weekday()
        .map(|ca| ca.into_column())
}
pub(super) fn day(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series().day().map(|ca| ca.into_column())
}
pub(super) fn ordinal_day(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .ordinal_day()
        .map(|ca| ca.into_column())
}
pub(super) fn time(s: &Column) -> PolarsResult<Column> {
    match s.dtype() {
        #[cfg(feature = "timezones")]
        DataType::Datetime(_, Some(_)) => polars_ops::prelude::replace_time_zone(
            s.datetime().unwrap(),
            None,
            &StringChunked::from_iter(std::iter::once("raise")),
            NonExistent::Raise,
        )?
        .cast(&DataType::Time)
        .map(Column::from),
        DataType::Datetime(_, _) => s
            .datetime()
            .unwrap()
            .cast(&DataType::Time)
            .map(Column::from),
        DataType::Time => Ok(s.clone()),
        dtype => polars_bail!(ComputeError: "expected Datetime or Time, got {}", dtype),
    }
}
pub(super) fn date(s: &Column) -> PolarsResult<Column> {
    match s.dtype() {
        #[cfg(feature = "timezones")]
        DataType::Datetime(_, Some(_)) => {
            let mut out = {
                polars_ops::chunked_array::replace_time_zone(
                    s.datetime().unwrap(),
                    None,
                    &StringChunked::from_iter(std::iter::once("raise")),
                    NonExistent::Raise,
                )?
                .cast(&DataType::Date)?
            };
            // `replace_time_zone` may unset sorted flag. But, we're only taking the date
            // part of the result, so we can safely preserve the sorted flag here. We may
            // need to make an exception if a time zone introduces a change which involves
            // "going back in time" and repeating a day, but we're not aware of that ever
            // having happened.
            out.set_sorted_flag(s.is_sorted_flag());
            Ok(out.into())
        },
        DataType::Datetime(_, _) => s
            .datetime()
            .unwrap()
            .cast(&DataType::Date)
            .map(Column::from),
        DataType::Date => Ok(s.clone()),
        dtype => polars_bail!(ComputeError: "expected Datetime or Date, got {}", dtype),
    }
}
pub(super) fn datetime(s: &Column) -> PolarsResult<Column> {
    match s.dtype() {
        #[cfg(feature = "timezones")]
        DataType::Datetime(tu, Some(_)) => polars_ops::chunked_array::replace_time_zone(
            s.datetime().unwrap(),
            None,
            &StringChunked::from_iter(std::iter::once("raise")),
            NonExistent::Raise,
        )?
        .cast(&DataType::Datetime(*tu, None))
        .map(|x| x.into()),
        DataType::Datetime(tu, _) => s
            .datetime()
            .unwrap()
            .cast(&DataType::Datetime(*tu, None))
            .map(Column::from),
        dtype => polars_bail!(ComputeError: "expected Datetime, got {}", dtype),
    }
}
pub(super) fn hour(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series().hour().map(|ca| ca.into_column())
}
pub(super) fn minute(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .minute()
        .map(|ca| ca.into_column())
}
pub(super) fn second(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .second()
        .map(|ca| ca.into_column())
}
pub(super) fn millisecond(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .nanosecond()
        .map(|ca| (ca.wrapping_trunc_div_scalar(1_000_000)).into_column())
}
pub(super) fn microsecond(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .nanosecond()
        .map(|ca| (ca.wrapping_trunc_div_scalar(1_000)).into_column())
}
pub(super) fn nanosecond(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .nanosecond()
        .map(|ca| ca.into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_days(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .duration()
        .map(|ca| ca.days().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_hours(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .duration()
        .map(|ca| ca.hours().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_minutes(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .duration()
        .map(|ca| ca.minutes().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_seconds(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .duration()
        .map(|ca| ca.seconds().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_milliseconds(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .duration()
        .map(|ca| ca.milliseconds().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_microseconds(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .duration()
        .map(|ca| ca.microseconds().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_nanoseconds(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .duration()
        .map(|ca| ca.nanoseconds().into_column())
}
pub(super) fn timestamp(s: &Column, tu: TimeUnit) -> PolarsResult<Column> {
    s.as_materialized_series()
        .timestamp(tu)
        .map(|ca| ca.into_column())
}
pub(super) fn to_string(s: &Column, format: &str) -> PolarsResult<Column> {
    TemporalMethods::to_string(s.as_materialized_series(), format).map(Column::from)
}

#[cfg(feature = "timezones")]
pub(super) fn convert_time_zone(s: &Column, time_zone: &TimeZone) -> PolarsResult<Column> {
    match s.dtype() {
        DataType::Datetime(_, _) => {
            let mut ca = s.datetime()?.clone();
            validate_time_zone(time_zone)?;
            ca.set_time_zone(time_zone.clone())?;
            Ok(ca.into_column())
        },
        dtype => polars_bail!(ComputeError: "expected Datetime, got {}", dtype),
    }
}
pub(super) fn with_time_unit(s: &Column, tu: TimeUnit) -> PolarsResult<Column> {
    match s.dtype() {
        DataType::Datetime(_, _) => {
            let mut ca = s.datetime()?.clone();
            ca.set_time_unit(tu);
            Ok(ca.into_column())
        },
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(_) => {
            let mut ca = s.as_materialized_series().duration()?.clone();
            ca.set_time_unit(tu);
            Ok(ca.into_column())
        },
        dt => polars_bail!(ComputeError: "dtype `{}` has no time unit", dt),
    }
}
pub(super) fn cast_time_unit(s: &Column, tu: TimeUnit) -> PolarsResult<Column> {
    match s.dtype() {
        DataType::Datetime(_, _) => {
            let ca = s.datetime()?;
            Ok(ca.cast_time_unit(tu).into_column())
        },
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(_) => {
            let ca = s.as_materialized_series().duration()?;
            Ok(ca.cast_time_unit(tu).into_column())
        },
        dt => polars_bail!(ComputeError: "dtype `{}` has no time unit", dt),
    }
}

pub(super) fn truncate(s: &[Column]) -> PolarsResult<Column> {
    let time_series = &s[0];
    let every = s[1].str()?;

    let mut out = match time_series.dtype() {
        DataType::Datetime(_, tz) => match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => time_series
                .datetime()?
                .truncate(tz.parse::<Tz>().ok().as_ref(), every)?
                .into_column(),
            _ => time_series.datetime()?.truncate(None, every)?.into_column(),
        },
        DataType::Date => time_series.date()?.truncate(None, every)?.into_column(),
        dt => polars_bail!(opq = round, got = dt, expected = "date/datetime"),
    };
    out.set_sorted_flag(time_series.is_sorted_flag());
    Ok(out)
}

#[cfg(feature = "offset_by")]
pub(super) fn offset_by(s: &[Column]) -> PolarsResult<Column> {
    impl_offset_by(s[0].as_materialized_series(), s[1].as_materialized_series()).map(Column::from)
}

#[cfg(feature = "month_start")]
pub(super) fn month_start(s: &Column) -> PolarsResult<Column> {
    Ok(match s.dtype() {
        DataType::Datetime(_, tz) => match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => s
                .datetime()
                .unwrap()
                .month_start(tz.parse::<Tz>().ok().as_ref())?
                .into_column(),
            _ => s.datetime().unwrap().month_start(None)?.into_column(),
        },
        DataType::Date => s.date().unwrap().month_start(None)?.into_column(),
        dt => polars_bail!(opq = month_start, got = dt, expected = "date/datetime"),
    })
}

#[cfg(feature = "month_end")]
pub(super) fn month_end(s: &Column) -> PolarsResult<Column> {
    Ok(match s.dtype() {
        DataType::Datetime(_, tz) => match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => s
                .datetime()
                .unwrap()
                .month_end(tz.parse::<Tz>().ok().as_ref())?
                .into_column(),
            _ => s.datetime().unwrap().month_end(None)?.into_column(),
        },
        DataType::Date => s.date().unwrap().month_end(None)?.into_column(),
        dt => polars_bail!(opq = month_end, got = dt, expected = "date/datetime"),
    })
}

#[cfg(feature = "timezones")]
pub(super) fn base_utc_offset(s: &Column) -> PolarsResult<Column> {
    match s.dtype() {
        DataType::Datetime(time_unit, Some(tz)) => {
            let tz = tz
                .parse::<Tz>()
                .expect("Time zone has already been validated");
            Ok(base_utc_offset_fn(s.datetime().unwrap(), time_unit, &tz).into_column())
        },
        dt => polars_bail!(
            opq = base_utc_offset,
            got = dt,
            expected = "time-zone-aware datetime"
        ),
    }
}
#[cfg(feature = "timezones")]
pub(super) fn dst_offset(s: &Column) -> PolarsResult<Column> {
    match s.dtype() {
        DataType::Datetime(time_unit, Some(tz)) => {
            let tz = tz
                .parse::<Tz>()
                .expect("Time zone has already been validated");
            Ok(dst_offset_fn(s.datetime().unwrap(), time_unit, &tz).into_column())
        },
        dt => polars_bail!(
            opq = dst_offset,
            got = dt,
            expected = "time-zone-aware datetime"
        ),
    }
}

pub(super) fn round(s: &[Column]) -> PolarsResult<Column> {
    let time_series = &s[0];
    let every = s[1].str()?;

    Ok(match time_series.dtype() {
        DataType::Datetime(_, tz) => match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => time_series
                .datetime()
                .unwrap()
                .round(every, tz.parse::<Tz>().ok().as_ref())?
                .into_column(),
            _ => time_series
                .datetime()
                .unwrap()
                .round(every, None)?
                .into_column(),
        },
        DataType::Date => time_series
            .date()
            .unwrap()
            .round(every, None)?
            .into_column(),
        dt => polars_bail!(opq = round, got = dt, expected = "date/datetime"),
    })
}

pub(super) fn replace(s: &[Column]) -> PolarsResult<Column> {
    let time_series = &s[0];
    let s_year = &s[1].strict_cast(&DataType::Int32)?;
    let s_month = &s[2].strict_cast(&DataType::Int8)?;
    let s_day = &s[3].strict_cast(&DataType::Int8)?;
    let year = s_year.i32()?;
    let month = s_month.i8()?;
    let day = s_day.i8()?;

    match time_series.dtype() {
        DataType::Datetime(_, _) => {
            let s_hour = &s[4].strict_cast(&DataType::Int8)?;
            let s_minute = &s[5].strict_cast(&DataType::Int8)?;
            let s_second = &s[6].strict_cast(&DataType::Int8)?;
            let s_microsecond = &s[7].strict_cast(&DataType::Int32)?;
            let hour = s_hour.i8()?;
            let minute = s_minute.i8()?;
            let second = s_second.i8()?;
            let nanosecond = &(s_microsecond.i32()? * 1_000);
            let s_ambiguous = &s[8].strict_cast(&DataType::String)?;
            let ambiguous = s_ambiguous.str()?;

            let out = replace_datetime(
                time_series.datetime().unwrap(),
                year,
                month,
                day,
                hour,
                minute,
                second,
                nanosecond,
                ambiguous,
            );
            out.map(|s| s.into_column())
        },
        DataType::Date => {
            let out = replace_date(time_series.date().unwrap(), year, month, day);
            out.map(|s| s.into_column())
        },
        dt => polars_bail!(opq = round, got = dt, expected = "date/datetime"),
    }
}
