use super::*;
use crate::{map, map_as_slice};

impl From<TemporalFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: TemporalFunction) -> Self {
        use TemporalFunction::*;
        match func {
            Millennium => map!(datetime::millennium),
            Century => map!(datetime::century),
            Year => map!(datetime::year),
            IsLeapYear => map!(datetime::is_leap_year),
            IsoYear => map!(datetime::iso_year),
            Month => map!(datetime::month),
            Quarter => map!(datetime::quarter),
            Week => map!(datetime::week),
            WeekDay => map!(datetime::weekday),
            Duration(tu) => map_as_slice!(impl_duration, tu),
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
            TotalDays => map!(datetime::total_days),
            TotalHours => map!(datetime::total_hours),
            TotalMinutes => map!(datetime::total_minutes),
            TotalSeconds => map!(datetime::total_seconds),
            TotalMilliseconds => map!(datetime::total_milliseconds),
            TotalMicroseconds => map!(datetime::total_microseconds),
            TotalNanoseconds => map!(datetime::total_nanoseconds),
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
            #[cfg(feature = "timezones")]
            ReplaceTimeZone(tz, non_existent) => {
                map_as_slice!(dispatch::replace_time_zone, tz.as_deref(), non_existent)
            },
            Combine(tu) => map_as_slice!(temporal::combine, tu),
            DatetimeFunction {
                time_unit,
                time_zone,
            } => {
                map_as_slice!(temporal::datetime, &time_unit, time_zone.as_deref())
            },
        }
    }
}

pub(super) fn datetime(
    s: &[Series],
    time_unit: &TimeUnit,
    time_zone: Option<&str>,
) -> PolarsResult<Series> {
    use polars_core::export::chrono::NaiveDate;
    use polars_core::utils::CustomIterTools;

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

    let mut month = month.cast(&DataType::UInt32)?;
    if month.len() < max_len {
        month = month.new_from_index(0, max_len);
    }
    let month = month.u32()?;

    let mut day = day.cast(&DataType::UInt32)?;
    if day.len() < max_len {
        day = day.new_from_index(0, max_len);
    }
    let day = day.u32()?;

    let mut hour = hour.cast(&DataType::UInt32)?;
    if hour.len() < max_len {
        hour = hour.new_from_index(0, max_len);
    }
    let hour = hour.u32()?;

    let mut minute = minute.cast(&DataType::UInt32)?;
    if minute.len() < max_len {
        minute = minute.new_from_index(0, max_len);
    }
    let minute = minute.u32()?;

    let mut second = second.cast(&DataType::UInt32)?;
    if second.len() < max_len {
        second = second.new_from_index(0, max_len);
    }
    let second = second.u32()?;

    let mut microsecond = microsecond.cast(&DataType::UInt32)?;
    if microsecond.len() < max_len {
        microsecond = microsecond.new_from_index(0, max_len);
    }
    let microsecond = microsecond.u32()?;
    let mut _ambiguous = ambiguous.cast(&DataType::String)?;
    if _ambiguous.len() < max_len {
        _ambiguous = _ambiguous.new_from_index(0, max_len);
    }
    let _ambiguous = _ambiguous.str()?;

    let ca: Int64Chunked = year
        .into_iter()
        .zip(month)
        .zip(day)
        .zip(hour)
        .zip(minute)
        .zip(second)
        .zip(microsecond)
        .map(|((((((y, m), d), h), mnt), s), us)| {
            if let (Some(y), Some(m), Some(d), Some(h), Some(mnt), Some(s), Some(us)) =
                (y, m, d, h, mnt, s, us)
            {
                NaiveDate::from_ymd_opt(y, m, d)
                    .and_then(|nd| nd.and_hms_micro_opt(h, mnt, s, us))
                    .map(|ndt| match time_unit {
                        TimeUnit::Milliseconds => ndt.and_utc().timestamp_millis(),
                        TimeUnit::Microseconds => ndt.and_utc().timestamp_micros(),
                        TimeUnit::Nanoseconds => ndt.and_utc().timestamp_nanos_opt().unwrap(),
                    })
            } else {
                None
            }
        })
        .collect_trusted();

    let ca = match time_zone {
        #[cfg(feature = "timezones")]
        Some(_) => {
            let mut ca = ca.into_datetime(*time_unit, None);
            ca = replace_time_zone(&ca, time_zone, _ambiguous, NonExistent::Raise)?;
            ca
        },
        _ => {
            polars_ensure!(
                time_zone.is_none(),
                ComputeError: "cannot make use of the `time_zone` argument without the 'timezones' feature enabled."
            );
            ca.into_datetime(*time_unit, None)
        },
    };

    let mut s = ca.into_series();
    s.rename("datetime");
    Ok(s)
}

pub(super) fn combine(s: &[Series], tu: TimeUnit) -> PolarsResult<Series> {
    let date = &s[0];
    let time = &s[1];

    let tz = match date.dtype() {
        DataType::Date => None,
        DataType::Datetime(_, tz) => tz.as_ref(),
        _dtype => {
            polars_bail!(ComputeError: format!("expected Date or Datetime, got {}", _dtype))
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
        .into()),
        _ => result_naive,
    }
}
