#[cfg(feature = "timezones")]
use arrow::legacy::time_zone::Tz;
#[cfg(feature = "timezones")]
use jiff::Timestamp;
#[cfg(feature = "timezones")]
use jiff::tz::{Dst, Offset};
#[cfg(feature = "timezones")]
use polars_core::prelude::*;

/// The "standard" (non-DST) offset for `tz` at the instant `ts`.
///
/// jiff doesn't expose a single "base UTC offset" query like `chrono-tz`'s
/// `OffsetComponents::base_utc_offset` does. If `ts` itself isn't in DST, its
/// offset already is the base offset. Otherwise, we look at the nearest
/// non-DST transition on either side of `ts` and pick whichever gives an
/// offset closest to the current one - DST adjustments are always small
/// (typically 30 minutes to 2 hours), whereas picking the "nearest in time"
/// transition can pick the wrong side of a rare *permanent* offset change
/// (e.g. Pacific/Apia's 2011 International Date Line shift, where the
/// chronologically-nearest non-DST transition can be on the wrong side of
/// the jump).
#[cfg(feature = "timezones")]
pub(crate) fn base_offset_at(tz: &Tz, ts: Timestamp) -> Offset {
    let info = tz.to_offset_info(ts);
    if matches!(info.dst(), Dst::No) {
        return info.offset();
    }
    let current = info.offset();
    let preceding = tz
        .preceding(ts)
        .find(|t| matches!(t.dst(), Dst::No))
        .map(|t| t.offset());
    let following = tz
        .following(ts)
        .find(|t| matches!(t.dst(), Dst::No))
        .map(|t| t.offset());
    match (preceding, following) {
        (Some(p), Some(f)) => {
            let p_delta = (current.seconds() - p.seconds()).abs();
            let f_delta = (current.seconds() - f.seconds()).abs();
            if p_delta <= f_delta { p } else { f }
        },
        (Some(p), None) => p,
        (None, Some(f)) => f,
        (None, None) => current,
    }
}

#[cfg(feature = "timezones")]
pub fn base_utc_offset(
    ca: &DatetimeChunked,
    time_unit: &TimeUnit,
    time_zone: &Tz,
) -> DurationChunked {
    ca.phys
        .apply_values(|t| {
            let ts = match time_unit {
                TimeUnit::Nanoseconds => Timestamp::from_nanosecond(t as i128),
                TimeUnit::Microseconds => Timestamp::from_microsecond(t),
                TimeUnit::Milliseconds => Timestamp::from_millisecond(t),
            }
            .expect("timestamp out-of-range");
            i64::from(base_offset_at(time_zone, ts).seconds()) * 1000
        })
        .into_duration(TimeUnit::Milliseconds)
}
