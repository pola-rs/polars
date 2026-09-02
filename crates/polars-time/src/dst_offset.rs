#[cfg(feature = "timezones")]
use arrow::legacy::time_zone::Tz;
#[cfg(feature = "timezones")]
use jiff::Timestamp;
#[cfg(feature = "timezones")]
use polars_core::prelude::*;

#[cfg(feature = "timezones")]
use crate::base_utc_offset::base_offset_at;

#[cfg(feature = "timezones")]
pub fn dst_offset(ca: &DatetimeChunked, time_unit: &TimeUnit, time_zone: &Tz) -> DurationChunked {
    ca.phys
        .apply_values(|t| {
            let ts = match time_unit {
                TimeUnit::Nanoseconds => Timestamp::from_nanosecond(t as i128),
                TimeUnit::Microseconds => Timestamp::from_microsecond(t),
                TimeUnit::Milliseconds => Timestamp::from_millisecond(t),
            }
            .expect("timestamp out-of-range");
            let current_offset = time_zone.to_offset_info(ts).offset();
            let base_offset = base_offset_at(time_zone, ts);
            i64::from(current_offset.seconds() - base_offset.seconds()) * 1000
        })
        .into_duration(TimeUnit::Milliseconds)
}
