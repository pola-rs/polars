#[cfg(feature = "timezones")]
use arrow::legacy::time_zone::Tz;
#[cfg(feature = "timezones")]
use chrono::TimeZone;
#[cfg(feature = "timezones")]
use chrono_tz::OffsetComponents;
#[cfg(feature = "timezones")]
use polars_core::prelude::*;
#[cfg(feature = "timezones")]
use polars_core::utils::arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};

#[cfg(feature = "timezones")]
pub fn dst_offset(ca: &DatetimeChunked, time_unit: &TimeUnit, time_zone: &Tz) -> DurationChunked {
    let timestamp_to_datetime = match time_unit {
        TimeUnit::Nanoseconds => timestamp_ns_to_datetime,
        TimeUnit::Microseconds => timestamp_us_to_datetime,
        TimeUnit::Milliseconds => timestamp_ms_to_datetime,
    };
    ca.0.apply_values(|t| {
        let ndt = timestamp_to_datetime(t);
        let dt = time_zone.from_utc_datetime(&ndt);
        dt.offset().dst_offset().num_milliseconds()
    })
    .into_duration(TimeUnit::Milliseconds)
}
