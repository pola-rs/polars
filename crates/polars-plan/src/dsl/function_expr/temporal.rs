#[cfg(feature = "date_offset")]
use polars_arrow::time_zone::Tz;
#[cfg(feature = "date_offset")]
use polars_time::prelude::*;

use super::*;

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
                        TimeUnit::Nanoseconds => ndt.timestamp_nanos(),
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
pub(super) fn date_offset(s: Series, offset: Duration) -> PolarsResult<Series> {
    let preserve_sortedness: bool;
    let out = match s.dtype().clone() {
        DataType::Date => {
            let s = s
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap();
            preserve_sortedness = true;
            date_offset(s, offset).and_then(|s| s.cast(&DataType::Date))
        },
        DataType::Datetime(tu, tz) => {
            let ca = s.datetime().unwrap();

            fn offset_fn(tu: TimeUnit) -> fn(&Duration, i64, Option<&Tz>) -> PolarsResult<i64> {
                match tu {
                    TimeUnit::Nanoseconds => Duration::add_ns,
                    TimeUnit::Microseconds => Duration::add_us,
                    TimeUnit::Milliseconds => Duration::add_ms,
                }
            }

            let out = match tz {
                #[cfg(feature = "timezones")]
                Some(ref tz) => {
                    let offset_fn = offset_fn(tu);
                    ca.0.try_apply(|v| offset_fn(&offset, v, tz.parse::<Tz>().ok().as_ref()))
                },
                _ => {
                    let offset_fn = offset_fn(tu);
                    ca.0.try_apply(|v| offset_fn(&offset, v, None))
                },
            }?;
            // Sortedness may not be preserved when crossing daylight savings time boundaries
            // for calendar-aware durations.
            // Constant durations (e.g. 2 hours) always preserve sortedness.
            preserve_sortedness =
                tz.is_none() || tz.as_deref() == Some("UTC") || offset.is_constant_duration();
            out.cast(&DataType::Datetime(tu, tz))
        },
        dt => polars_bail!(
            ComputeError: "cannot use 'date_offset' on Series of datatype {}", dt,
        ),
    };
    if preserve_sortedness {
        out.map(|mut out| {
            out.set_sorted_flag(s.is_sorted_flag());
            out
        })
    } else {
        out
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
