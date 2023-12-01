use std::str::FromStr;

use arrow::legacy::kernels::{convert_to_naive_local, Ambiguous};
use arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};
use chrono::NaiveDateTime;
use chrono_tz::{Tz, UTC};
use polars_core::chunked_array::ops::arity::try_binary_elementwise;
use polars_core::prelude::*;

fn parse_time_zone(s: &str) -> PolarsResult<Tz> {
    s.parse()
        .map_err(|e| polars_err!(ComputeError: format!("unable to parse time zone: '{s}': {e}")))
}

pub fn replace_time_zone(
    datetime: &Logical<DatetimeType, Int64Type>,
    time_zone: Option<&str>,
    ambiguous: &Utf8Chunked,
) -> PolarsResult<DatetimeChunked> {
    let from_time_zone = datetime.time_zone().as_deref().unwrap_or("UTC");
    let from_tz = parse_time_zone(from_time_zone)?;
    let to_tz = parse_time_zone(time_zone.unwrap_or("UTC"))?;
    if (from_tz == to_tz)
        & ((from_tz == UTC)
            | ((ambiguous.len() == 1) & (unsafe { ambiguous.get_unchecked(0) } == Some("raise"))))
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
    let out = match ambiguous.len() {
        1 => match unsafe { ambiguous.get_unchecked(0) } {
            Some(ambiguous) => datetime.0.try_apply(|timestamp| {
                let ndt = timestamp_to_datetime(timestamp);
                Ok(datetime_to_timestamp(convert_to_naive_local(
                    &from_tz,
                    &to_tz,
                    ndt,
                    Ambiguous::from_str(ambiguous)?,
                )?))
            }),
            _ => Ok(datetime.0.apply(|_| None)),
        },
        _ => try_binary_elementwise(datetime, ambiguous, |timestamp_opt, ambiguous_opt| {
            match (timestamp_opt, ambiguous_opt) {
                (Some(timestamp), Some(ambiguous)) => {
                    let ndt = timestamp_to_datetime(timestamp);
                    Ok(Some(datetime_to_timestamp(convert_to_naive_local(
                        &from_tz,
                        &to_tz,
                        ndt,
                        Ambiguous::from_str(ambiguous)?,
                    )?)))
                },
                _ => Ok(None),
            }
        }),
    };
    let mut out = out?.into_datetime(datetime.time_unit(), time_zone.map(|x| x.to_string()));
    if from_time_zone == "UTC" && ambiguous.len() == 1 && ambiguous.get(0).unwrap() == "raise" {
        // In general, the sortedness flag can't be preserved.
        // To be safe, we only do so in the simplest case when we know for sure that there is no "daylight savings weirdness" going on, i.e.:
        // - `from_tz` is guaranteed to not observe daylight savings time;
        // - user is just passing 'raise' to 'ambiguous'.
        // Both conditions above need to be satisfied.
        out.set_sorted_flag(datetime.is_sorted_flag());
    }
    Ok(out)
}

pub fn convert_and_replace_time_zone(
    datetime: &Logical<DatetimeType, Int64Type>,
    replace_tz: Option<&str>,
    convert_tz: &Utf8Chunked,
) -> PolarsResult<DatetimeChunked> {
    let from_time_zone = datetime.time_zone().as_deref().unwrap_or("UTC");
    let from_tz = parse_time_zone(from_time_zone)?;

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
    let out = match convert_tz.len() {
        1 => match unsafe { convert_tz.get_unchecked(0) } {
            Some(convert_tz) => datetime.0.try_apply(|timestamp| {
                let ndt = timestamp_to_datetime(timestamp);
                Ok(datetime_to_timestamp(convert_to_naive_local(
                    &parse_time_zone(convert_tz)?,
                    &from_tz,
                    ndt,
                    Ambiguous::Raise,
                )?))
            }),
            _ => Ok(datetime.0.apply(|_| None)),
        },
        _ => try_binary_elementwise(datetime, convert_tz, |timestamp_opt, ambiguous_opt| match (
            timestamp_opt,
            ambiguous_opt,
        ) {
            (Some(timestamp), Some(convert_tz)) => {
                let ndt = timestamp_to_datetime(timestamp);
                Ok(Some(datetime_to_timestamp(convert_to_naive_local(
                    &parse_time_zone(convert_tz)?,
                    &from_tz,
                    ndt,
                    Ambiguous::Raise,
                )?)))
            },
            _ => Ok(None),
        }),
    };
    let out = out?.into_datetime(datetime.time_unit(), replace_tz.map(|x| x.to_string()));
    Ok(out)
}
