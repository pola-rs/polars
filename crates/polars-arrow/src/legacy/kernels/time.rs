use std::str::FromStr;

#[cfg(feature = "timezones")]
use chrono::{LocalResult, NaiveDateTime, TimeZone};
#[cfg(feature = "timezones")]
use chrono_tz::Tz;
#[cfg(feature = "timezones")]
use polars_error::PolarsResult;
use polars_error::{polars_bail, PolarsError};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub enum Ambiguous {
    Earliest,
    Latest,
    Null,
    Raise,
}
impl FromStr for Ambiguous {
    type Err = PolarsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "earliest" => Ok(Ambiguous::Earliest),
            "latest" => Ok(Ambiguous::Latest),
            "raise" => Ok(Ambiguous::Raise),
            "null" => Ok(Ambiguous::Null),
            s => polars_bail!(InvalidOperation:
                "Invalid argument {}, expected one of: \"earliest\", \"latest\", \"null\", \"raise\"", s
            ),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NonExistent {
    Null,
    Raise,
}

#[cfg(feature = "timezones")]
pub fn convert_to_naive_local(
    from_tz: &Tz,
    to_tz: &Tz,
    ndt: NaiveDateTime,
    ambiguous: Ambiguous,
    non_existent: NonExistent,
) -> PolarsResult<Option<NaiveDateTime>> {
    let ndt = from_tz.from_utc_datetime(&ndt).naive_local();
    match to_tz.from_local_datetime(&ndt) {
        LocalResult::Single(dt) => Ok(Some(dt.naive_utc())),
        LocalResult::Ambiguous(dt_earliest, dt_latest) => match ambiguous {
            Ambiguous::Earliest => Ok(Some(dt_earliest.naive_utc())),
            Ambiguous::Latest => Ok(Some(dt_latest.naive_utc())),
            Ambiguous::Null => Ok(None),
            Ambiguous::Raise => {
                polars_bail!(ComputeError: "datetime '{}' is ambiguous in time zone '{}'. Please use `ambiguous` to tell how it should be localized.", ndt, to_tz)
            },
        },
        LocalResult::None => match non_existent {
            NonExistent::Raise => polars_bail!(ComputeError:
                "datetime '{}' is non-existent in time zone '{}'. You may be able to use `non_existent='null'` to return `null` in this case.",
                ndt, to_tz
            ),
            NonExistent::Null => Ok(None),
        },
    }
}

/// Same as convert_to_naive_local, but return `None` instead
/// raising - in some cases this can be used to save a string allocation.
#[cfg(feature = "timezones")]
pub fn convert_to_naive_local_opt(
    from_tz: &Tz,
    to_tz: &Tz,
    ndt: NaiveDateTime,
    ambiguous: Ambiguous,
) -> Option<Option<NaiveDateTime>> {
    let ndt = from_tz.from_utc_datetime(&ndt).naive_local();
    match to_tz.from_local_datetime(&ndt) {
        LocalResult::Single(dt) => Some(Some(dt.naive_utc())),
        LocalResult::Ambiguous(dt_earliest, dt_latest) => match ambiguous {
            Ambiguous::Earliest => Some(Some(dt_earliest.naive_utc())),
            Ambiguous::Latest => Some(Some(dt_latest.naive_utc())),
            Ambiguous::Null => Some(None),
            Ambiguous::Raise => None,
        },
        LocalResult::None => None,
    }
}
