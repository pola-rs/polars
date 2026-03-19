#[cfg(feature = "timezones")]
use arrow::legacy::time_zone::Tz;
use polars_core::error::{PolarsResult, polars_bail};
use polars_core::prelude::{
    ArithmeticChunked, Column, DataType, IntoColumn, LogicalType, TimeUnit,
};
#[cfg(feature = "timezones")]
use polars_core::prelude::{NonExistent, StringChunked, TimeZone};
use polars_time::prelude::*;
use polars_time::replace_datetime;
use polars_time::series::TemporalMethods;

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
pub(super) fn days_in_month(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .days_in_month()
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
        dtype => polars_bail!(ComputeError: "expected Datetime or Time, got {dtype}"),
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
        dtype => polars_bail!(ComputeError: "expected Datetime or Date, got {dtype}"),
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
        dtype => polars_bail!(ComputeError: "expected Datetime, got {dtype}"),
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
    use polars_time::prelude::DurationMethods;

    s.as_materialized_series()
        .duration()
        .map(|ca| ca.days().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_days_fractional(s: &Column) -> PolarsResult<Column> {
    use polars_time::prelude::DurationMethods;

    s.as_materialized_series()
        .duration()
        .map(|ca| ca.days_fractional().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_hours(s: &Column) -> PolarsResult<Column> {
    use polars_time::prelude::DurationMethods;

    s.as_materialized_series()
        .duration()
        .map(|ca| ca.hours().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_hours_fractional(s: &Column) -> PolarsResult<Column> {
    use polars_time::prelude::DurationMethods;

    s.as_materialized_series()
        .duration()
        .map(|ca| ca.hours_fractional().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_minutes(s: &Column) -> PolarsResult<Column> {
    use polars_time::prelude::DurationMethods;

    s.as_materialized_series()
        .duration()
        .map(|ca| ca.minutes().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_minutes_fractional(s: &Column) -> PolarsResult<Column> {
    use polars_time::prelude::DurationMethods;

    s.as_materialized_series()
        .duration()
        .map(|ca| ca.minutes_fractional().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_seconds(s: &Column) -> PolarsResult<Column> {
    use polars_time::prelude::DurationMethods;

    s.as_materialized_series()
        .duration()
        .map(|ca| ca.seconds().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_seconds_fractional(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .duration()
        .map(|ca| ca.seconds_fractional().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_milliseconds(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .duration()
        .map(|ca| ca.milliseconds().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_milliseconds_fractional(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .duration()
        .map(|ca| ca.milliseconds_fractional().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_microseconds(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .duration()
        .map(|ca| ca.microseconds().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_microseconds_fractional(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .duration()
        .map(|ca| ca.microseconds_fractional().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_nanoseconds(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .duration()
        .map(|ca| ca.nanoseconds().into_column())
}
#[cfg(feature = "dtype-duration")]
pub(super) fn total_nanoseconds_fractional(s: &Column) -> PolarsResult<Column> {
    s.as_materialized_series()
        .duration()
        .map(|ca| ca.nanoseconds_fractional().into_column())
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
            ca.set_time_zone(time_zone.clone())?;
            Ok(ca.into_column())
        },
        dtype => polars_bail!(ComputeError: "expected Datetime, got {dtype}"),
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
    use polars_time::impl_offset_by;

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
            Ok(polars_time::base_utc_offset(s.datetime().unwrap(), time_unit, &tz).into_column())
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
            Ok(polars_time::dst_offset(s.datetime().unwrap(), time_unit, &tz).into_column())
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
            let out = polars_time::replace_date(time_series.date().unwrap(), year, month, day);
            out.map(|s| s.into_column())
        },
        dt => polars_bail!(opq = round, got = dt, expected = "date/datetime"),
    }
}
