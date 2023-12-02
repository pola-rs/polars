#[cfg(feature = "date_offset")]
use arrow::legacy::time_zone::Tz;
#[cfg(feature = "date_offset")]
use polars_core::chunked_array::ops::arity::try_binary_elementwise;
#[cfg(feature = "date_offset")]
use polars_time::prelude::*;

use super::*;
use crate::{map, map_as_slice};

impl From<TemporalFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: TemporalFunction) -> Self {
        use TemporalFunction::*;
        match func {
            Year => map!(datetime::year),
            IsLeapYear => map!(datetime::is_leap_year),
            IsoYear => map!(datetime::iso_year),
            Month => map!(datetime::month),
            Quarter => map!(datetime::quarter),
            Week => map!(datetime::week),
            WeekDay => map!(datetime::weekday),
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
            ToString(format) => map!(datetime::to_string, &format),
            TimeStamp(tu) => map!(datetime::timestamp, tu),
            #[cfg(feature = "timezones")]
            ConvertTimeZone(tz) => map!(datetime::convert_time_zone, &tz),
            WithTimeUnit(tu) => map!(datetime::with_time_unit, tu),
            CastTimeUnit(tu) => map!(datetime::cast_time_unit, tu),
            Truncate(offset) => {
                map_as_slice!(datetime::truncate, &offset)
            },
            #[cfg(feature = "date_offset")]
            MonthStart => map!(datetime::month_start),
            #[cfg(feature = "date_offset")]
            MonthEnd => map!(datetime::month_end),
            #[cfg(feature = "timezones")]
            BaseUtcOffset => map!(datetime::base_utc_offset),
            #[cfg(feature = "timezones")]
            DSTOffset => map!(datetime::dst_offset),
            Round(every, offset) => map_as_slice!(datetime::round, &every, &offset),
            #[cfg(feature = "timezones")]
            ReplaceTimeZone(tz) => {
                map_as_slice!(dispatch::replace_time_zone, tz.as_deref())
            },
            ConvertToLocalTimeZone => {
                map_as_slice!(dispatch::convert_to_local_time_zone)
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
    let mut _ambiguous = ambiguous.cast(&DataType::Utf8)?;
    if _ambiguous.len() < max_len {
        _ambiguous = _ambiguous.new_from_index(0, max_len);
    }
    let _ambiguous = _ambiguous.utf8()?;

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
                        TimeUnit::Milliseconds => ndt.timestamp_millis(),
                        TimeUnit::Microseconds => ndt.timestamp_micros(),
                        TimeUnit::Nanoseconds => ndt.timestamp_nanos_opt().unwrap(),
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
            ca = replace_time_zone(&ca, time_zone, _ambiguous)?;
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

#[cfg(feature = "date_offset")]
fn apply_offsets_to_datetime(
    datetime: &Logical<DatetimeType, Int64Type>,
    offsets: &Utf8Chunked,
    offset_fn: fn(&Duration, i64, Option<&Tz>) -> PolarsResult<i64>,
    time_zone: Option<&Tz>,
) -> PolarsResult<Int64Chunked> {
    match (datetime.len(), offsets.len()) {
        (1, _) => match datetime.0.get(0) {
            Some(dt) => offsets.try_apply_values_generic(|offset| {
                offset_fn(&Duration::parse(offset), dt, time_zone)
            }),
            _ => Ok(Int64Chunked::full_null(datetime.0.name(), offsets.len())),
        },
        (_, 1) => match offsets.get(0) {
            Some(offset) => datetime
                .0
                .try_apply(|v| offset_fn(&Duration::parse(offset), v, time_zone)),
            _ => Ok(datetime.0.apply(|_| None)),
        },
        _ => try_binary_elementwise(datetime, offsets, |timestamp_opt, offset_opt| {
            match (timestamp_opt, offset_opt) {
                (Some(timestamp), Some(offset)) => {
                    offset_fn(&Duration::parse(offset), timestamp, time_zone).map(Some)
                },
                _ => Ok(None),
            }
        }),
    }
}

#[cfg(feature = "date_offset")]
pub(super) fn date_offset(s: &[Series]) -> PolarsResult<Series> {
    let ts = &s[0];
    let offsets = &s[1].utf8().unwrap();

    let preserve_sortedness: bool;
    let out = match ts.dtype() {
        DataType::Date => {
            let ts = ts
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap();
            let datetime = ts.datetime().unwrap();
            let out = apply_offsets_to_datetime(datetime, offsets, Duration::add_ms, None)?;
            // sortedness is only guaranteed to be preserved if a constant offset is being added to every datetime
            preserve_sortedness = match offsets.len() {
                1 => offsets.get(0).is_some(),
                _ => false,
            };
            out.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap()
                .cast(&DataType::Date)
        },
        DataType::Datetime(tu, tz) => {
            let datetime = ts.datetime().unwrap();

            let offset_fn = match tu {
                TimeUnit::Nanoseconds => Duration::add_ns,
                TimeUnit::Microseconds => Duration::add_us,
                TimeUnit::Milliseconds => Duration::add_ms,
            };

            let out = match tz {
                #[cfg(feature = "timezones")]
                Some(ref tz) => apply_offsets_to_datetime(
                    datetime,
                    offsets,
                    offset_fn,
                    tz.parse::<Tz>().ok().as_ref(),
                )?,
                _ => apply_offsets_to_datetime(datetime, offsets, offset_fn, None)?,
            };
            // Sortedness may not be preserved when crossing daylight savings time boundaries
            // for calendar-aware durations.
            // Constant durations (e.g. 2 hours) always preserve sortedness.
            preserve_sortedness = match offsets.len() {
                1 => match offsets.get(0) {
                    Some(offset) => {
                        let offset = Duration::parse(offset);
                        tz.is_none()
                            || tz.as_deref() == Some("UTC")
                            || offset.is_constant_duration()
                    },
                    None => false,
                },
                _ => false,
            };
            out.cast(&DataType::Datetime(*tu, tz.clone()))
        },
        dt => polars_bail!(
            ComputeError: "cannot use 'date_offset' on Series of datatype {}", dt,
        ),
    };
    if preserve_sortedness {
        out.map(|mut out| {
            out.set_sorted_flag(ts.is_sorted_flag());
            out
        })
    } else {
        out.map(|mut out| {
            out.set_sorted_flag(IsSorted::Not);
            out
        })
    }
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
            result_naive.datetime().unwrap(),
            Some(tz),
            &Utf8Chunked::from_iter(std::iter::once("raise")),
        )?
        .into()),
        _ => Ok(result_naive),
    }
}
