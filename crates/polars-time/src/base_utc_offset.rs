#[cfg(feature = "timezones")]
use chrono::TimeZone;
#[cfg(feature = "timezones")]
use chrono_tz::OffsetComponents;
#[cfg(feature = "timezones")]
use polars_arrow::time_zone::Tz;
#[cfg(feature = "timezones")]
use polars_core::prelude::*;

#[cfg(feature = "timezones")]
pub fn base_utc_offset(
    ca: &DatetimeChunked,
    time_unit: &TimeUnit,
    time_zone: &Tz,
) -> DurationChunked {
    let timestamp_to_datetime = timestamp_to_naive_datetime_method(time_unit);
    ca.0.apply_values(|t| {
        let ndt = timestamp_to_datetime(t);
        let dt = time_zone.from_utc_datetime(&ndt);
        dt.offset().base_utc_offset().num_milliseconds()
    })
    .into_duration(TimeUnit::Milliseconds)
}
