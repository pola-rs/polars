#[cfg(feature = "timezones")]
use chrono::TimeZone;
#[cfg(feature = "timezones")]
use chrono_tz::OffsetComponents;
#[cfg(feature = "timezones")]
use polars_arrow::time_zone::Tz;
#[cfg(feature = "timezones")]
use polars_core::prelude::*;
#[cfg(feature = "timezones")]
use polars_core::utils::arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};

#[cfg(feature = "timezones")]
pub trait PolarsDSTOffset {
    fn dst_offset(&self, time_zone: &Tz) -> DurationChunked
    where
        Self: Sized;
}

#[cfg(feature = "timezones")]
impl PolarsDSTOffset for DatetimeChunked {
    fn dst_offset(&self, time_zone: &Tz) -> DurationChunked {
        let timestamp_to_datetime = match self.time_unit() {
            TimeUnit::Nanoseconds => timestamp_ns_to_datetime,
            TimeUnit::Microseconds => timestamp_us_to_datetime,
            TimeUnit::Milliseconds => timestamp_ms_to_datetime,
        };
        self.0
            .apply(|t| {
                let ndt = timestamp_to_datetime(t);
                let dt = time_zone.from_utc_datetime(&ndt);
                dt.offset().dst_offset().num_milliseconds()
            })
            .into_duration(TimeUnit::Milliseconds)
    }
}
