//! Traits and utilities for temporal data.
pub mod conversion;
#[cfg(feature = "dtype-date")]
mod date;
#[cfg(feature = "dtype-datetime")]
mod datetime;
#[cfg(feature = "dtype-duration")]
mod duration;
#[cfg(feature = "dtype-time")]
mod time;
#[cfg(feature = "dtype-date")]
use chrono::NaiveDate;
use chrono::NaiveDateTime;
#[cfg(any(feature = "dtype-time", feature = "dtype-date"))]
use chrono::NaiveTime;
#[cfg(feature = "timezones")]
use chrono_tz::Tz;
#[cfg(feature = "timezones")]
use once_cell::sync::Lazy;
#[cfg(all(feature = "regex", feature = "timezones"))]
use regex::Regex;
#[cfg(feature = "dtype-time")]
pub use time::time_to_time64ns;

pub use self::conversion::*;
#[cfg(feature = "timezones")]
use crate::prelude::{polars_bail, PolarsResult};

#[cfg(feature = "timezones")]
static FIXED_OFFSET_PATTERN: &str = r#"(?x)
    ^
    (?P<sign>[-+])?            # optional sign
    (?P<hour>0[0-9]|1[0-4])    # hour (between 0 and 14)
    :?                         # optional separator
    00                         # minute
    $
    "#;
#[cfg(feature = "timezones")]
static FIXED_OFFSET_RE: Lazy<Regex> = Lazy::new(|| Regex::new(FIXED_OFFSET_PATTERN).unwrap());

#[cfg(feature = "timezones")]
pub fn validate_time_zone(tz: &str) -> PolarsResult<()> {
    match tz.parse::<Tz>() {
        Ok(_) => Ok(()),
        Err(_) => {
            polars_bail!(ComputeError: "unable to parse time zone: '{}'. Please check the Time Zone Database for a list of available time zones", tz)
        },
    }
}

#[cfg(feature = "timezones")]
pub fn parse_time_zone(tz: &str) -> PolarsResult<Tz> {
    match tz.parse::<Tz>() {
        Ok(tz) => Ok(tz),
        Err(_) => {
            polars_bail!(ComputeError: "unable to parse time zone: '{}'. Please check the Time Zone Database for a list of available time zones", tz)
        },
    }
}

/// Convert fixed offset to Etc/GMT one from time zone database
///
/// E.g. +01:00 -> Etc/GMT-1
///
/// Note: the sign appears reversed, but is correct, see https://en.wikipedia.org/wiki/Tz_database#Area:
/// > In order to conform with the POSIX style, those zone names beginning with
/// > "Etc/GMT" have their sign reversed from the standard ISO 8601 convention.
/// > In the "Etc" area, zones west of GMT have a positive sign and those east
/// > have a negative sign in their name (e.g "Etc/GMT-14" is 14 hours ahead of GMT).
#[cfg(feature = "timezones")]
pub fn parse_fixed_offset(tz: &str) -> PolarsResult<String> {
    if let Some(caps) = FIXED_OFFSET_RE.captures(tz) {
        let sign = match caps.name("sign").map(|s| s.as_str()) {
            Some("-") => "+",
            _ => "-",
        };
        let hour = caps.name("hour").unwrap().as_str().parse::<i32>().unwrap();
        let etc_tz = format!("Etc/GMT{}{}", sign, hour);
        if etc_tz.parse::<Tz>().is_ok() {
            return Ok(etc_tz);
        }
    }
    polars_bail!(ComputeError: "unable to parse time zone: '{}'. Please check the Time Zone Database for a list of available time zones", tz)
}
