use std::str::FromStr;

use arrow::legacy::kernels::convert_to_naive_local;
use arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};
use chrono::NaiveDateTime;
use chrono_tz::UTC;
use polars_core::chunked_array::ops::arity::try_binary_elementwise;
use polars_core::chunked_array::temporal::parse_time_zone;
use polars_core::prelude::*;

pub fn replace_time_zone(
    datetime: &Logical<DatetimeType, Int64Type>,
    time_zone: Option<&str>,
    ambiguous: &StringChunked,
    non_existent: NonExistent,
) -> PolarsResult<DatetimeChunked> {
    let from_time_zone = datetime.time_zone().as_deref().unwrap_or("UTC");
    let from_tz = parse_time_zone(from_time_zone)?;
    let to_tz = parse_time_zone(time_zone.unwrap_or("UTC"))?;
    if (from_tz == to_tz)
        & ((from_tz == UTC) | ((ambiguous.len() == 1) & (ambiguous.get(0) == Some("raise"))))
    {
        let mut out = datetime
            .0
            .clone()
            .into_datetime(datetime.time_unit(), time_zone.map(|x| x.to_string()));
        out.set_sorted_flag(datetime.is_sorted_flag());
        return Ok(out);
    }
    let timestamp_to_datetime: fn(i64) -> NaiveDateTime = match datetime.time_unit() {
        TimeUnit::Milliseconds => timestamp_ms_to_datetime,
        TimeUnit::Microseconds => timestamp_us_to_datetime,
        TimeUnit::Nanoseconds => timestamp_ns_to_datetime,
    };
    let datetime_to_timestamp: fn(NaiveDateTime) -> i64 = match datetime.time_unit() {
        TimeUnit::Milliseconds => datetime_to_timestamp_ms,
        TimeUnit::Microseconds => datetime_to_timestamp_us,
        TimeUnit::Nanoseconds => datetime_to_timestamp_ns,
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

    let mut out = out?.into_datetime(datetime.time_unit(), time_zone.map(|x| x.to_string()));
    if from_time_zone == "UTC" && ambiguous.len() == 1 && ambiguous.get(0) == Some("raise") {
        // In general, the sortedness flag can't be preserved.
        // To be safe, we only do so in the simplest case when we know for sure that there is no "daylight savings weirdness" going on, i.e.:
        // - `from_tz` is guaranteed to not observe daylight savings time;
        // - user is just passing 'raise' to 'ambiguous'.
        // Both conditions above need to be satisfied.
        out.set_sorted_flag(datetime.is_sorted_flag());
    }
    Ok(out)
}

/// If `ambiguous` is length-1 and not equal to "null", we can take a slightly faster path.
pub fn impl_replace_time_zone_fast(
    datetime: &Logical<DatetimeType, Int64Type>,
    ambiguous: Option<&str>,
    timestamp_to_datetime: fn(i64) -> NaiveDateTime,
    datetime_to_timestamp: fn(NaiveDateTime) -> i64,
    from_tz: &chrono_tz::Tz,
    to_tz: &chrono_tz::Tz,
) -> PolarsResult<Int64Chunked> {
    match ambiguous {
        Some(ambiguous) => datetime.0.try_apply_nonnull_values_generic(|timestamp| {
            let ndt = timestamp_to_datetime(timestamp);
            Ok(datetime_to_timestamp(
                convert_to_naive_local(
                    from_tz,
                    to_tz,
                    ndt,
                    Ambiguous::from_str(ambiguous)?,
                    NonExistent::Raise,
                )?
                .expect("we didn't use Ambiguous::Null or NonExistent::Null"),
            ))
        }),
        _ => Ok(datetime.0.apply(|_| None)),
    }
}

pub fn impl_replace_time_zone(
    datetime: &Logical<DatetimeType, Int64Type>,
    ambiguous: &StringChunked,
    non_existent: NonExistent,
    timestamp_to_datetime: fn(i64) -> NaiveDateTime,
    datetime_to_timestamp: fn(NaiveDateTime) -> i64,
    from_tz: &chrono_tz::Tz,
    to_tz: &chrono_tz::Tz,
) -> PolarsResult<Int64Chunked> {
    match ambiguous.len() {
        1 => {
            let iter = datetime.0.downcast_iter().map(|arr| {
                let element_iter = arr.iter().map(|timestamp_opt| match timestamp_opt {
                    Some(timestamp) => {
                        let ndt = timestamp_to_datetime(*timestamp);
                        let res = convert_to_naive_local(
                            from_tz,
                            to_tz,
                            ndt,
                            Ambiguous::from_str(ambiguous.get(0).unwrap())?,
                            non_existent,
                        )?;
                        Ok::<_, PolarsError>(res.map(datetime_to_timestamp))
                    },
                    None => Ok(None),
                });
                element_iter.try_collect_arr()
            });
            ChunkedArray::try_from_chunk_iter(datetime.0.name(), iter)
        },
        _ => try_binary_elementwise(datetime, ambiguous, |timestamp_opt, ambiguous_opt| {
            match (timestamp_opt, ambiguous_opt) {
                (Some(timestamp), Some(ambiguous)) => {
                    let ndt = timestamp_to_datetime(timestamp);
                    Ok(convert_to_naive_local(
                        from_tz,
                        to_tz,
                        ndt,
                        Ambiguous::from_str(ambiguous)?,
                        non_existent,
                    )?
                    .map(datetime_to_timestamp))
                },
                _ => Ok(None),
            }
        }),
    }
}
