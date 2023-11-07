use std::str::FromStr;

#[cfg(feature = "timezones")]
use chrono::{LocalResult, NaiveDateTime, TimeZone};
#[cfg(feature = "timezones")]
use chrono_tz::Tz;
#[cfg(feature = "timezones")]
use polars_error::PolarsResult;
use polars_error::{polars_bail, PolarsError};

pub enum Ambiguous {
    Earliest,
    Latest,
    Raise,
}
impl FromStr for Ambiguous {
    type Err = PolarsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "earliest" => Ok(Ambiguous::Earliest),
            "latest" => Ok(Ambiguous::Latest),
            "raise" => Ok(Ambiguous::Raise),
            s => polars_bail!(InvalidOperation:
                "Invalid argument {}, expected one of: \"earliest\", \"latest\", \"raise\"", s
            ),
        }
    }
}

#[cfg(feature = "timezones")]
pub fn convert_to_naive_local(
    from_tz: &Tz,
    to_tz: &Tz,
    ndt: NaiveDateTime,
    ambiguous: Ambiguous,
) -> PolarsResult<NaiveDateTime> {
    let ndt = from_tz.from_utc_datetime(&ndt).naive_local();
    match to_tz.from_local_datetime(&ndt) {
        LocalResult::Single(dt) => Ok(dt.naive_utc()),
        LocalResult::Ambiguous(dt_earliest, dt_latest) => match ambiguous {
            Ambiguous::Earliest => Ok(dt_earliest.naive_utc()),
            Ambiguous::Latest => Ok(dt_latest.naive_utc()),
            Ambiguous::Raise => {
                polars_bail!(ComputeError: "datetime '{}' is ambiguous in time zone '{}'. Please use `ambiguous` to tell how it should be localized.", ndt, to_tz)
            },
        },
        LocalResult::None => polars_bail!(ComputeError:
                "datetime '{}' is non-existent in time zone '{}'. Non-existent datetimes are not yet supported",
                ndt, to_tz
        ),
    }
}

#[cfg(feature = "timezones")]
pub fn convert_to_naive_local_opt(
    from_tz: &Tz,
    to_tz: &Tz,
    ndt: NaiveDateTime,
    ambiguous: Ambiguous,
) -> Option<NaiveDateTime> {
    let ndt = from_tz.from_utc_datetime(&ndt).naive_local();
    match to_tz.from_local_datetime(&ndt) {
        LocalResult::Single(dt) => Some(dt.naive_utc()),
        LocalResult::Ambiguous(dt_earliest, dt_latest) => match ambiguous {
            Ambiguous::Earliest => Some(dt_earliest.naive_utc()),
            Ambiguous::Latest => Some(dt_latest.naive_utc()),
            Ambiguous::Raise => None,
        },
        LocalResult::None => None,
    }
}
