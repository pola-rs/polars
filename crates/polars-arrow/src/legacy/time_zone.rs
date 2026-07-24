// a placeholder type for when timezones are not enabled
#[cfg(not(feature = "timezones"))]
#[derive(Copy, Clone)]
pub enum Tz {}
#[cfg(feature = "timezones")]
pub use jiff::tz::TimeZone as Tz;
