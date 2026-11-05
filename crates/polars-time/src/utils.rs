#[cfg(feature = "timezones")]
use chrono_tz::TZ_VARIANTS;

#[cfg(feature = "timezones")]
pub fn known_timezones() -> [&'static str; TZ_VARIANTS.len()] {
    core::array::from_fn(|i| TZ_VARIANTS[i].name())
}
