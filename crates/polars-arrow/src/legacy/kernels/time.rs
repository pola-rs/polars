use chrono::{LocalResult, NaiveDateTime, TimeZone};
use chrono_tz::Tz;
use polars_error::{polars_bail, PolarsResult};

pub fn convert_to_naive_local(
    from_tz: &Tz,
    to_tz: &Tz,
    ndt: NaiveDateTime,
    ambiguous: &str,
) -> PolarsResult<NaiveDateTime> {
    let ndt = from_tz.from_utc_datetime(&ndt).naive_local();
    match to_tz.from_local_datetime(&ndt) {
        LocalResult::Single(dt) => Ok(dt.naive_utc()),
        LocalResult::Ambiguous(dt_earliest, dt_latest) => match ambiguous {
            "earliest" => Ok(dt_earliest.naive_utc()),
            "latest" => Ok(dt_latest.naive_utc()),
            "raise" => {
                polars_bail!(InvalidOperation: "datetime '{}' is ambiguous in time zone '{}'. Please use `ambiguous` to tell how it should be localized.", ndt, to_tz)
            },
            ambiguous => polars_bail!(InvalidOperation:
                "Invalid argument {}, expected one of: \"earliest\", \"latest\", \"raise\"", ambiguous
            ),
        },
        LocalResult::None => polars_bail!(InvalidOperation:
                "datetime '{}' is non-existent in time zone '{}'. Non-existent datetimes are not yet supported",
                ndt, to_tz
        ),
    }
}
