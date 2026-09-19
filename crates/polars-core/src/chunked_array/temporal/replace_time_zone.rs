use std::str::FromStr;

use chrono::NaiveDateTime;
use chrono_tz::UTC;
use polars_arrow::legacy::kernels::convert_to_naive_local;
use polars_arrow::temporal_conversions::{
    timestamp_ms_to_datetime_opt, timestamp_ns_to_datetime_opt, timestamp_us_to_datetime_opt,
};

use crate::chunked_array::ops::arity::try_binary_elementwise;
use crate::prelude::*;

pub fn replace_time_zone(
    datetime: &Logical<DatetimeType, Int64Type>,
    time_zone: Option<&TimeZone>,
    ambiguous: &StringChunked,
    non_existent: NonExistent,
) -> PolarsResult<DatetimeChunked> {
    let from_time_zone = datetime.time_zone().clone().unwrap_or(TimeZone::UTC);

    let from_tz = from_time_zone.to_chrono()?;

    let to_tz = if let Some(tz) = time_zone {
        tz.to_chrono()?
    } else {
        chrono_tz::UTC
    };

    if (from_tz == to_tz)
        & ((from_tz == UTC) | ((ambiguous.len() == 1) & (ambiguous.get(0) == Some("raise"))))
    {
        let mut out = datetime
            .phys
            .clone()
            .into_datetime(datetime.time_unit(), time_zone.cloned());
        out.physical_mut()
            .set_sorted_flag(datetime.physical().is_sorted_flag());
        return Ok(out);
    }
    let timestamp_to_datetime: fn(i64) -> Option<NaiveDateTime> = match datetime.time_unit() {
        TimeUnit::Milliseconds => timestamp_ms_to_datetime_opt,
        TimeUnit::Microseconds => timestamp_us_to_datetime_opt,
        TimeUnit::Nanoseconds => timestamp_ns_to_datetime_opt,
    };
    let datetime_to_timestamp: fn(NaiveDateTime) -> Option<i64> = match datetime.time_unit() {
        TimeUnit::Milliseconds => |dt| Some(datetime_to_timestamp_ms(dt)),
        TimeUnit::Microseconds => |dt| Some(datetime_to_timestamp_us(dt)),
        TimeUnit::Nanoseconds => |dt| dt.and_utc().timestamp_nanos_opt(),
    };

    let out = if ambiguous.len() == 1
        && ambiguous.get(0) != Some("null")
        && non_existent == NonExistent::Raise
    {
        impl_replace_time_zone_fast(
            datetime,
            ambiguous.get(0),
            timestamp_to_datetime,
            datetime_to_timestamp,
            &from_tz,
            &to_tz,
        )
    } else {
        impl_replace_time_zone(
            datetime,
            ambiguous,
            non_existent,
            timestamp_to_datetime,
            datetime_to_timestamp,
            &from_tz,
            &to_tz,
        )
    };

    let mut out = out?.into_datetime(datetime.time_unit(), time_zone.cloned());
    if from_time_zone == TimeZone::UTC && ambiguous.len() == 1 && ambiguous.get(0) == Some("raise")
    {
        // In general, the sortedness flag can't be preserved.
        // To be safe, we only do so in the simplest case when we know for sure that there is no "daylight savings weirdness" going on, i.e.:
        // - `from_tz` is guaranteed to not observe daylight savings time;
        // - user is just passing 'raise' to 'ambiguous'.
        // Both conditions above need to be satisfied.
        out.physical_mut()
            .set_sorted_flag(datetime.physical().is_sorted_flag());
    }
    Ok(out)
}

/// If `ambiguous` is length-1 and not equal to "null", we can take a slightly faster path.
pub fn impl_replace_time_zone_fast(
    datetime: &Logical<DatetimeType, Int64Type>,
    ambiguous: Option<&str>,
    timestamp_to_datetime: fn(i64) -> Option<NaiveDateTime>,
    datetime_to_timestamp: fn(NaiveDateTime) -> Option<i64>,
    from_tz: &chrono_tz::Tz,
    to_tz: &chrono_tz::Tz,
) -> PolarsResult<Int64Chunked> {
    match ambiguous {
        Some(ambiguous) => datetime.phys.try_apply_nonnull_values_generic(|timestamp| {
            let ndt = timestamp_to_datetime(timestamp).ok_or_else(out_of_range)?;
            datetime_to_timestamp(
                convert_to_naive_local(
                    from_tz,
                    to_tz,
                    ndt,
                    Ambiguous::from_str(ambiguous)?,
                    NonExistent::Raise,
                )?
                .expect("we didn't use Ambiguous::Null or NonExistent::Null"),
            )
            .ok_or_else(out_of_range)
        }),
        _ => Ok(datetime.phys.apply(|_| None)),
    }
}

pub fn impl_replace_time_zone(
    datetime: &Logical<DatetimeType, Int64Type>,
    ambiguous: &StringChunked,
    non_existent: NonExistent,
    timestamp_to_datetime: fn(i64) -> Option<NaiveDateTime>,
    datetime_to_timestamp: fn(NaiveDateTime) -> Option<i64>,
    from_tz: &chrono_tz::Tz,
    to_tz: &chrono_tz::Tz,
) -> PolarsResult<Int64Chunked> {
    match ambiguous.len() {
        1 => {
            let Some(ambiguous) = ambiguous.get(0) else {
                return Ok(datetime.phys.apply(|_| None));
            };
            let iter = datetime.phys.downcast_iter().map(|arr| {
                let element_iter = arr.iter().map(|timestamp_opt| match timestamp_opt {
                    Some(timestamp) => {
                        let ndt = timestamp_to_datetime(*timestamp).ok_or_else(out_of_range)?;
                        let res = convert_to_naive_local(
                            from_tz,
                            to_tz,
                            ndt,
                            Ambiguous::from_str(ambiguous)?,
                            non_existent,
                        )?;
                        res.map(|dt| datetime_to_timestamp(dt).ok_or_else(out_of_range))
                            .transpose()
                    },
                    None => Ok(None),
                });
                element_iter.try_collect_arr()
            });
            ChunkedArray::try_from_chunk_iter(datetime.phys.name().clone(), iter)
        },
        _ => try_binary_elementwise(
            datetime.physical(),
            ambiguous,
            |timestamp_opt, ambiguous_opt| match (timestamp_opt, ambiguous_opt) {
                (Some(timestamp), Some(ambiguous)) => {
                    let ndt = timestamp_to_datetime(timestamp).ok_or_else(out_of_range)?;
                    convert_to_naive_local(
                        from_tz,
                        to_tz,
                        ndt,
                        Ambiguous::from_str(ambiguous)?,
                        non_existent,
                    )?
                    .map(|dt| datetime_to_timestamp(dt).ok_or_else(out_of_range))
                    .transpose()
                },
                _ => Ok(None),
            },
        ),
    }
}

fn out_of_range() -> PolarsError {
    polars_err!(ComputeError: "datetime is out of range for time zone conversion")
}
