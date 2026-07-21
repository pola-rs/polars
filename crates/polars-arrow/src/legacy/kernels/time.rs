use std::str::FromStr;

#[cfg(feature = "timezones")]
use jiff::civil::DateTime;
#[cfg(feature = "timezones")]
use jiff::tz::{AmbiguousOffset, TimeZone};
#[cfg(feature = "timezones")]
use polars_error::PolarsResult;
use polars_error::{PolarsError, polars_bail};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

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

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[strum(serialize_all = "snake_case")]
pub enum NonExistent {
    Null,
    Raise,
}

/// Reinterprets `ndt` (a naive/civil datetime that represents a physical instant,
/// UTC-labelled) as a wall-clock time in `from_tz`, then reinterprets that same
/// wall-clock time as belonging to `to_tz`, resolving any resulting ambiguity
/// (DST fold) or non-existence (DST gap) per `ambiguous`/`non_existent`.
#[cfg(feature = "timezones")]
pub fn convert_to_naive_local(
    from_tz: &TimeZone,
    to_tz: &TimeZone,
    ndt: DateTime,
    ambiguous: Ambiguous,
    non_existent: NonExistent,
) -> PolarsResult<Option<DateTime>> {
    let ts = TimeZone::UTC.to_timestamp(ndt).map_err(|_| {
        PolarsError::ComputeError(format!("datetime '{}' is out-of-range", fmt_ndt(ndt)).into())
    })?;
    let ndt = from_tz.to_datetime(ts);
    let ambiguous_ts = to_tz.to_ambiguous_timestamp(ndt);
    let offset = match ambiguous_ts.offset() {
        AmbiguousOffset::Unambiguous { offset } => offset,
        AmbiguousOffset::Fold { before, after } => match ambiguous {
            Ambiguous::Earliest => before,
            Ambiguous::Latest => after,
            Ambiguous::Null => return Ok(None),
            Ambiguous::Raise => {
                polars_bail!(ComputeError: "datetime '{}' is ambiguous in time zone '{}'. Please use `ambiguous` to tell how it should be localized.", fmt_ndt(ndt), to_tz.iana_name().unwrap_or("<unknown>"))
            },
        },
        AmbiguousOffset::Gap { .. } => match non_existent {
            NonExistent::Raise => polars_bail!(ComputeError:
                "datetime '{}' is non-existent in time zone '{}'. You may be able to use `non_existent='null'` to return `null` in this case.",
                fmt_ndt(ndt), to_tz.iana_name().unwrap_or("<unknown>")
            ),
            NonExistent::Null => return Ok(None),
        },
    };
    let final_ts = offset.to_timestamp(ndt).map_err(|_| {
        PolarsError::ComputeError(format!("datetime '{}' is out-of-range", fmt_ndt(ndt)).into())
    })?;
    Ok(Some(TimeZone::UTC.to_datetime(final_ts)))
}

/// Formats a civil datetime the way chrono's `NaiveDateTime` `Display` did
/// (space-separated, not jiff's default ISO 8601 "T" separator), since
/// downstream error messages and tests depend on this exact shape.
#[cfg(feature = "timezones")]
fn fmt_ndt(dt: DateTime) -> impl std::fmt::Display {
    dt.strftime("%Y-%m-%d %H:%M:%S%.f")
}

/// Same as convert_to_naive_local, but return `None` instead
/// raising - in some cases this can be used to save a string allocation.
#[cfg(feature = "timezones")]
pub fn convert_to_naive_local_opt(
    from_tz: &TimeZone,
    to_tz: &TimeZone,
    ndt: DateTime,
    ambiguous: Ambiguous,
) -> Option<Option<DateTime>> {
    let ts = TimeZone::UTC.to_timestamp(ndt).ok()?;
    let ndt = from_tz.to_datetime(ts);
    let ambiguous_ts = to_tz.to_ambiguous_timestamp(ndt);
    let offset = match ambiguous_ts.offset() {
        AmbiguousOffset::Unambiguous { offset } => offset,
        AmbiguousOffset::Fold { before, after } => match ambiguous {
            Ambiguous::Earliest => before,
            Ambiguous::Latest => after,
            Ambiguous::Null => return Some(None),
            Ambiguous::Raise => return None,
        },
        AmbiguousOffset::Gap { .. } => return None,
    };
    let final_ts = offset.to_timestamp(ndt).ok()?;
    Some(Some(TimeZone::UTC.to_datetime(final_ts)))
}

#[cfg(all(test, feature = "timezones"))]
mod tests {
    use jiff::civil::date;

    use super::*;

    // `convert_to_naive_local(from_tz, to_tz, ndt, ..)` reinterprets `ndt` (a
    // UTC-labelled instant when `from_tz` is UTC) as a wall-clock reading and
    // asks what instant that same wall-clock reading corresponds to in
    // `to_tz` -- i.e. the core of `replace_time_zone`.

    #[test]
    fn test_convert_to_naive_local_unambiguous() {
        let utc = TimeZone::UTC;
        let ny = TimeZone::get("America/New_York").unwrap();
        // Wall-clock "2024-01-15 12:00:00" in America/New_York (EST, -05:00)
        // is the instant 2024-01-15T17:00:00Z.
        let ndt = date(2024, 1, 15).at(12, 0, 0, 0);
        let result =
            convert_to_naive_local(&utc, &ny, ndt, Ambiguous::Raise, NonExistent::Raise).unwrap();
        assert_eq!(result, Some(date(2024, 1, 15).at(17, 0, 0, 0)));
    }

    #[test]
    fn test_convert_to_naive_local_fold() {
        // Fall-back transition: on 2024-11-03, America/New_York repeats the
        // wall-clock hour 01:00-01:59, once in EDT (-04:00) and once in EST (-05:00).
        let utc = TimeZone::UTC;
        let ny = TimeZone::get("America/New_York").unwrap();
        let ambiguous_ndt = date(2024, 11, 3).at(1, 30, 0, 0);

        let earliest = convert_to_naive_local(
            &utc,
            &ny,
            ambiguous_ndt,
            Ambiguous::Earliest,
            NonExistent::Raise,
        )
        .unwrap();
        assert_eq!(earliest, Some(date(2024, 11, 3).at(5, 30, 0, 0)));

        let latest = convert_to_naive_local(
            &utc,
            &ny,
            ambiguous_ndt,
            Ambiguous::Latest,
            NonExistent::Raise,
        )
        .unwrap();
        assert_eq!(latest, Some(date(2024, 11, 3).at(6, 30, 0, 0)));

        let null = convert_to_naive_local(
            &utc,
            &ny,
            ambiguous_ndt,
            Ambiguous::Null,
            NonExistent::Raise,
        )
        .unwrap();
        assert_eq!(null, None);

        let err = convert_to_naive_local(
            &utc,
            &ny,
            ambiguous_ndt,
            Ambiguous::Raise,
            NonExistent::Raise,
        );
        assert!(err.is_err());
    }

    #[test]
    fn test_convert_to_naive_local_gap() {
        // Spring-forward transition: on 2024-03-10, America/New_York clocks jump from
        // 01:59:59 straight to 03:00:00, so 02:00-02:59 does not exist.
        let utc = TimeZone::UTC;
        let ny = TimeZone::get("America/New_York").unwrap();
        let gap_ndt = date(2024, 3, 10).at(2, 30, 0, 0);

        let result =
            convert_to_naive_local(&utc, &ny, gap_ndt, Ambiguous::Raise, NonExistent::Null)
                .unwrap();
        assert_eq!(result, None);

        let result =
            convert_to_naive_local(&utc, &ny, gap_ndt, Ambiguous::Raise, NonExistent::Raise);
        assert!(result.is_err());
    }
}
