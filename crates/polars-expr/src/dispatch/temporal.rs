use polars_core::chunked_array::ops::arity::{
    binary_elementwise_values, broadcast_try_binary_elementwise, unary_elementwise_values,
};
use polars_core::error::{PolarsResult, polars_bail};
use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;
use polars_defs::time::date_diff::DateDiffUnit;
use polars_plan::plans::IRTemporalFunction;
use polars_utils::broadcast::broadcast_len;
use polars_utils::pl_str::PlSmallStr;

use super::*;

pub fn temporal_func_to_udf(func: IRTemporalFunction) -> SpecialEq<Arc<dyn ColumnsUdf>> {
    use IRTemporalFunction::*;
    match func {
        Millennium => map!(datetime::millennium),
        Century => map!(datetime::century),
        Year => map!(datetime::year),
        IsLeapYear => map!(datetime::is_leap_year),
        IsoYear => map!(datetime::iso_year),
        Month => map!(datetime::month),
        DaysInMonth => map!(datetime::days_in_month),
        Quarter => map!(datetime::quarter),
        Week => map!(datetime::week),
        WeekDay => map!(datetime::weekday),
        #[cfg(feature = "dtype-duration")]
        Duration(tu) => map_as_slice!(polars_ops::series::impl_duration, tu),
        Day => map!(datetime::day),
        OrdinalDay => map!(datetime::ordinal_day),
        Time => map!(datetime::time),
        Date => map!(datetime::date),
        Hour => map!(datetime::hour),
        Minute => map!(datetime::minute),
        Second => map!(datetime::second),
        Millisecond => map!(datetime::millisecond),
        Microsecond => map!(datetime::microsecond),
        Nanosecond => map!(datetime::nanosecond),
        #[cfg(feature = "dtype-duration")]
        TotalDays { fractional: false } => map!(datetime::total_days),
        #[cfg(feature = "dtype-duration")]
        TotalDays { fractional: true } => map!(datetime::total_days_fractional),
        #[cfg(feature = "dtype-duration")]
        TotalHours { fractional: false } => map!(datetime::total_hours),
        #[cfg(feature = "dtype-duration")]
        TotalHours { fractional: true } => map!(datetime::total_hours_fractional),
        #[cfg(feature = "dtype-duration")]
        TotalMinutes { fractional: false } => map!(datetime::total_minutes),
        #[cfg(feature = "dtype-duration")]
        TotalMinutes { fractional: true } => map!(datetime::total_minutes_fractional),
        #[cfg(feature = "dtype-duration")]
        TotalSeconds { fractional: false } => map!(datetime::total_seconds),
        #[cfg(feature = "dtype-duration")]
        TotalSeconds { fractional: true } => map!(datetime::total_seconds_fractional),
        #[cfg(feature = "dtype-duration")]
        TotalMilliseconds { fractional: false } => map!(datetime::total_milliseconds),
        #[cfg(feature = "dtype-duration")]
        TotalMilliseconds { fractional: true } => map!(datetime::total_milliseconds_fractional),
        #[cfg(feature = "dtype-duration")]
        TotalMicroseconds { fractional: false } => map!(datetime::total_microseconds),
        #[cfg(feature = "dtype-duration")]
        TotalMicroseconds { fractional: true } => map!(datetime::total_microseconds_fractional),
        #[cfg(feature = "dtype-duration")]
        TotalNanoseconds { fractional: false } => map!(datetime::total_nanoseconds),
        #[cfg(feature = "dtype-duration")]
        TotalNanoseconds { fractional: true } => map!(datetime::total_nanoseconds_fractional),
        ToString(format) => map!(datetime::to_string, &format),
        TimeStamp(tu) => map!(datetime::timestamp, tu),
        #[cfg(feature = "timezones")]
        ConvertTimeZone(tz) => map!(datetime::convert_time_zone, &tz),
        CastTimeUnit(tu) => map!(datetime::cast_time_unit, tu),
        Truncate => {
            map_as_slice!(datetime::truncate)
        },
        #[cfg(feature = "offset_by")]
        OffsetBy => {
            map_as_slice!(datetime::offset_by)
        },
        #[cfg(feature = "month_start")]
        MonthStart => map!(datetime::month_start),
        #[cfg(feature = "month_end")]
        MonthEnd => map!(datetime::month_end),
        #[cfg(feature = "timezones")]
        BaseUtcOffset => map!(datetime::base_utc_offset),
        #[cfg(feature = "timezones")]
        DSTOffset => map!(datetime::dst_offset),
        Round => map_as_slice!(datetime::round),
        Replace => map_as_slice!(datetime::replace),
        #[cfg(feature = "timezones")]
        ReplaceTimeZone(tz, non_existent) => {
            map_as_slice!(misc::replace_time_zone, tz.as_ref(), non_existent)
        },
        Combine(tu) => map_as_slice!(temporal::combine, tu),
        DatetimeFunction {
            time_unit,
            time_zone,
        } => {
            map_as_slice!(temporal::datetime, &time_unit, time_zone.as_ref())
        },
        DateDiff(unit) => map_as_slice!(temporal::date_diff, unit),
    }
}

#[cfg(feature = "dtype-datetime")]
pub(super) fn datetime(
    s: &[Column],
    time_unit: &TimeUnit,
    time_zone: Option<&TimeZone>,
) -> PolarsResult<Column> {
    use polars_core::prelude::{DataType, DatetimeChunked};
    use polars_time::prelude::DatetimeMethods;

    let col_name = PlSmallStr::from_static("datetime");

    if s.iter().any(|s| s.is_empty()) {
        return Ok(Column::new_empty(
            col_name,
            &DataType::Datetime(
                time_unit.to_owned(),
                match time_zone.cloned() {
                    #[cfg(feature = "timezones")]
                    Some(v) => Some(v),
                    _ => {
                        assert!(
                            time_zone.is_none(),
                            "cannot make use of the `time_zone` argument without the 'timezones' feature enabled."
                        );
                        None
                    },
                },
            ),
        ));
    }

    let year = &s[0];
    let month = &s[1];
    let day = &s[2];
    let hour = &s[3];
    let minute = &s[4];
    let second = &s[5];
    let microsecond = &s[6];
    let ambiguous = &s[7];

    let len = broadcast_len(s.iter())?;

    let year = year.cast(&DataType::Int32)?.broadcast_owned_to(len)?;
    let year = year.i32()?;

    let month = month.cast(&DataType::Int8)?.broadcast_owned_to(len)?;
    let month = month.i8()?;

    let day = day.cast(&DataType::Int8)?.broadcast_owned_to(len)?;
    let day = day.i8()?;

    let hour = hour.cast(&DataType::Int8)?.broadcast_owned_to(len)?;
    let hour = hour.i8()?;

    let minute = minute.cast(&DataType::Int8)?.broadcast_owned_to(len)?;
    let minute = minute.i8()?;

    let second = second.cast(&DataType::Int8)?.broadcast_owned_to(len)?;
    let second = second.i8()?;

    let nanosecond = (microsecond.cast(&DataType::Int32)? * 1_000).broadcast_owned_to(len)?;
    let nanosecond = nanosecond.i32()?;

    let ambiguous = ambiguous.cast(&DataType::String)?.broadcast_owned_to(len)?;
    let ambiguous = ambiguous.str()?;

    let ca = DatetimeChunked::new_from_parts(
        year,
        month,
        day,
        hour,
        minute,
        second,
        nanosecond,
        ambiguous,
        time_unit,
        time_zone.cloned(),
        col_name,
    );
    ca.map(|s| s.into_column())
}

pub(super) fn combine(s: &[Column], tu: TimeUnit) -> PolarsResult<Column> {
    let date = &s[0];
    let time = &s[1];

    let tz = match date.dtype() {
        DataType::Date => None,
        DataType::Datetime(_, tz) => tz.as_ref(),
        dtype => {
            polars_bail!(ComputeError: "expected Date or Datetime, got {dtype}")
        },
    };

    let date = date.cast(&DataType::Date)?;
    let datetime = date.cast(&DataType::Datetime(tu, None)).unwrap();

    let duration = time.cast(&DataType::Duration(tu))?;
    let result_naive = datetime + duration;
    match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => Ok(
            polars_core::chunked_array::temporal::replace_time_zone::replace_time_zone(
                result_naive?.datetime().unwrap(),
                Some(tz),
                &StringChunked::from_iter(std::iter::once("raise")),
                NonExistent::Raise,
            )?
            .into_column(),
        ),
        _ => result_naive,
    }
}

pub(super) fn date_diff(s: &[Column], unit: DateDiffUnit) -> PolarsResult<Column> {
    let start = &s[0];
    let end = &s[1];
    polars_ensure!(
        start.len() == end.len() || start.len() == 1 || end.len() == 1,
        length_mismatch = "date_diff",
        start.len(),
        end.len()
    );
    let mut dtype = try_get_supertype(start.dtype(), end.dtype())?;
    // Use the finest time unit of the inputs, so that no boundaries are lost.
    if let DataType::Datetime(tu, _) = &mut dtype {
        for input in [start.dtype(), end.dtype()] {
            if let DataType::Datetime(input_tu, _) = input
                && tick_nanoseconds(*input_tu) < tick_nanoseconds(*tu)
            {
                *tu = *input_tu;
            }
        }
    }
    let (start_index, factor) = date_diff_index(&start.strict_cast(&dtype)?, unit)?;
    let (end_index, _) = date_diff_index(&end.strict_cast(&dtype)?, unit)?;
    let out: Int64Chunked = broadcast_try_binary_elementwise(&start_index, &end_index, |s, e| {
        let (Some(s), Some(e)) = (s, e) else {
            return Ok(None);
        };
        e.checked_sub(s)
            .and_then(|diff| diff.checked_mul(factor))
            .map(Some)
            .ok_or_else(|| polars_err!(ComputeError: "overflow in `date_diff`"))
    })?;
    Ok(out.with_name(start.name().clone()).into_column())
}

/// Number the `unit` boundaries of each value, such that `date_diff` is the difference
/// of two numbers multiplied by the returned factor.
fn date_diff_index(c: &Column, unit: DateDiffUnit) -> PolarsResult<(Int64Chunked, i64)> {
    use polars_arrow::temporal_conversions::NANOSECONDS_IN_DAY;

    let index = match (c.dtype(), unit.nanoseconds()) {
        (DataType::Null, _) => Int64Chunked::full_null(c.name().clone(), c.len()),
        (DataType::Date | DataType::Datetime(_, _), None) => calendar_index(c, unit)?,
        (DataType::Date, Some(ns)) => {
            let days = unary_elementwise_values(c.date()?.physical(), i64::from);
            return Ok((days, NANOSECONDS_IN_DAY / ns));
        },
        (DataType::Datetime(tu, tz), Some(ns)) => {
            let ca = c.datetime()?;
            let tick_ns = tick_nanoseconds(*tu);
            if ns <= tick_ns {
                return Ok((ca.physical().clone(), tick_ns / ns));
            }
            let k = ns / tick_ns;
            if tz.is_some() {
                // Truncate in wall-clock time, keeping the UTC offset of the value.
                binary_elementwise_values(ca.physical(), &wall_clock(ca)?, |t, local| {
                    t.div_euclid(k) - i64::from(t.rem_euclid(k) < local.rem_euclid(k))
                })
            } else {
                ca.physical().apply_values(|t| t.div_euclid(k))
            }
        },
        (DataType::Time, Some(ns)) => c
            .as_materialized_series()
            .time()?
            .physical()
            .apply_values(|t| t.div_euclid(ns)),
        (DataType::Time, None) => polars_bail!(
            InvalidOperation: "`date_diff` with unit '{}' is not supported for Time",
            <&str>::from(unit)
        ),
        (dtype, _) => polars_bail!(
            InvalidOperation: "`date_diff` expects Date, Datetime or Time inputs, got {dtype}"
        ),
    };
    Ok((index, 1))
}

fn tick_nanoseconds(tu: TimeUnit) -> i64 {
    match tu {
        TimeUnit::Nanoseconds => 1,
        TimeUnit::Microseconds => 1_000,
        TimeUnit::Milliseconds => 1_000_000,
    }
}

/// Physical values of the wall-clock time; the values themselves without a time zone.
fn wall_clock(ca: &DatetimeChunked) -> PolarsResult<Int64Chunked> {
    match ca.time_zone() {
        #[cfg(feature = "timezones")]
        Some(_) => Ok(
            polars_core::chunked_array::temporal::replace_time_zone::replace_time_zone(
                ca,
                None,
                &StringChunked::from_iter(std::iter::once("raise")),
                NonExistent::Raise,
            )?
            .physical()
            .clone(),
        ),
        _ => Ok(ca.physical().clone()),
    }
}

/// Number of the day, week, month, quarter or year of each date or wall-clock date.
fn calendar_index(c: &Column, unit: DateDiffUnit) -> PolarsResult<Int64Chunked> {
    use polars_arrow::temporal_conversions::NANOSECONDS_IN_DAY;
    use polars_time::prelude::DateMethods;

    let days = || -> PolarsResult<Int64Chunked> {
        Ok(match c.dtype() {
            DataType::Datetime(tu, _) => {
                let ticks_in_day = NANOSECONDS_IN_DAY / tick_nanoseconds(*tu);
                wall_clock(c.datetime()?)?.apply_values(|t| t.div_euclid(ticks_in_day))
            },
            _ => unary_elementwise_values(c.date()?.physical(), i64::from),
        })
    };
    match unit {
        DateDiffUnit::Day => return days(),
        // 1970-01-01 is a Thursday; weeks start on Monday.
        DateDiffUnit::Week => return Ok(days()?.apply_values(|d| (d + 3).div_euclid(7))),
        _ => {},
    }
    let dates = datetime::date(c)?;
    let dates = dates.date()?;
    let years = dates.year();
    Ok(match unit {
        DateDiffUnit::Year => unary_elementwise_values(&years, i64::from),
        DateDiffUnit::Quarter => {
            binary_elementwise_values(&years, &dates.month(), |y: i32, m: i8| {
                i64::from(y) * 4 + (i64::from(m) - 1) / 3
            })
        },
        _ => binary_elementwise_values(&years, &dates.month(), |y: i32, m: i8| {
            i64::from(y) * 12 + i64::from(m) - 1
        }),
    })
}
