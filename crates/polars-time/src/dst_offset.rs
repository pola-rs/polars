#[cfg(feature = "timezones")]
use chrono::TimeZone;
#[cfg(feature = "timezones")]
use chrono_tz::OffsetComponents;
#[cfg(feature = "timezones")]
use polars_arrow::legacy::time_zone::Tz;
#[cfg(feature = "timezones")]
use polars_core::prelude::*;

#[cfg(feature = "timezones")]
pub fn dst_offset(ca: &DatetimeChunked, time_unit: &TimeUnit, time_zone: &Tz) -> DurationChunked {
    ca.phys
        .apply_values(|t| {
            let ndt = time_unit.timestamp_to_datetime(t);
            let dt = time_zone.from_utc_datetime(&ndt);
            dt.offset().dst_offset().num_milliseconds()
        })
        .into_duration(TimeUnit::Milliseconds)
}
