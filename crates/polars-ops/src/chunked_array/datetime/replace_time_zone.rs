use chrono_tz::{Tz, UTC};
use polars_arrow::kernels::convert_to_naive_local;
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
    let timestamp_to_datetime = timestamp_to_naive_datetime_method(&datetime.time_unit());
    let datetime_to_timestamp = datetime_to_timestamp_method(&datetime.time_unit());

    let out = match ambiguous.len() {
        1 => match unsafe { ambiguous.get_unchecked(0) } {
            Some(ambiguous) => datetime.0.try_apply(|timestamp| {
                let ndt = timestamp_to_datetime(timestamp);
                Ok(datetime_to_timestamp(convert_to_naive_local(
                    &from_tz, &to_tz, ndt, ambiguous,
                )?))
            }),
            _ => Ok(datetime.0.apply(|_| None)),
        },
        _ => try_binary_elementwise(datetime, ambiguous, |timestamp_opt, ambiguous_opt| {
            match (timestamp_opt, ambiguous_opt) {
                (Some(timestamp), Some(ambiguous)) => {
                    let ndt = timestamp_to_datetime(timestamp);
                    Ok(Some(datetime_to_timestamp(convert_to_naive_local(
                        &from_tz, &to_tz, ndt, ambiguous,
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
