use polars_core::error::{PolarsResult, polars_bail};
use polars_core::prelude::*;
use polars_plan::plans::IRTemporalFunction;
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
        Datetime => map!(datetime::datetime),
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
        WithTimeUnit(tu) => map!(datetime::with_time_unit, tu),
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

    let max_len = s.iter().map(|s| s.len()).max().unwrap();

    let mut year = year.cast(&DataType::Int32)?;
    if year.len() < max_len {
        year = year.new_from_index(0, max_len)
    }
    let year = year.i32()?;

    let mut month = month.cast(&DataType::Int8)?;
    if month.len() < max_len {
        month = month.new_from_index(0, max_len);
    }
    let month = month.i8()?;

    let mut day = day.cast(&DataType::Int8)?;
    if day.len() < max_len {
        day = day.new_from_index(0, max_len);
    }
    let day = day.i8()?;

    let mut hour = hour.cast(&DataType::Int8)?;
    if hour.len() < max_len {
        hour = hour.new_from_index(0, max_len);
    }
    let hour = hour.i8()?;

    let mut minute = minute.cast(&DataType::Int8)?;
    if minute.len() < max_len {
        minute = minute.new_from_index(0, max_len);
    }
    let minute = minute.i8()?;

    let mut second = second.cast(&DataType::Int8)?;
    if second.len() < max_len {
        second = second.new_from_index(0, max_len);
    }
    let second = second.i8()?;

    let mut nanosecond = microsecond.cast(&DataType::Int32)? * 1_000;
    if nanosecond.len() < max_len {
        nanosecond = nanosecond.new_from_index(0, max_len);
    }
    let nanosecond = nanosecond.i32()?;

    let mut _ambiguous = ambiguous.cast(&DataType::String)?;
    if _ambiguous.len() < max_len {
        _ambiguous = _ambiguous.new_from_index(0, max_len);
    }
    let ambiguous = _ambiguous.str()?;

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
        Some(tz) => Ok(polars_ops::prelude::replace_time_zone(
            result_naive?.datetime().unwrap(),
            Some(tz),
            &StringChunked::from_iter(std::iter::once("raise")),
            NonExistent::Raise,
        )?
        .into_column()),
        _ => result_naive,
    }
}
