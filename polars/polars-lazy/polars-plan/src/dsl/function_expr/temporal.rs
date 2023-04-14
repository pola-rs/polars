#[cfg(feature = "timezones")]
use arrow::temporal_conversions::parse_offset;
#[cfg(feature = "timezones")]
use chrono_tz::Tz;
#[cfg(feature = "date_offset")]
use polars_arrow::time_zone::PolarsTimeZone;
#[cfg(feature = "date_offset")]
use polars_time::prelude::*;

use super::*;

#[cfg(feature = "date_offset")]
pub(super) fn date_offset(s: Series, offset: Duration) -> PolarsResult<Series> {
    match s.dtype().clone() {
        DataType::Date => {
            let s = s
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap();
            date_offset(s, offset).and_then(|s| s.cast(&DataType::Date))
        }
        DataType::Datetime(tu, tz) => {
            let ca = s.datetime().unwrap();

            fn adder<T: PolarsTimeZone>(
                tu: TimeUnit,
            ) -> fn(&Duration, i64, Option<&T>) -> PolarsResult<i64> {
                match tu {
                    TimeUnit::Nanoseconds => Duration::add_ns,
                    TimeUnit::Microseconds => Duration::add_us,
                    TimeUnit::Milliseconds => Duration::add_ms,
                }
            }

            let out = match tz {
                #[cfg(feature = "timezones")]
                Some(ref tz) => match tz.parse::<Tz>() {
                    Ok(tz) => ca.0.try_apply(|v| adder(tu)(&offset, v, Some(&tz))),
                    Err(_) => match parse_offset(tz) {
                        Ok(tz) => ca.0.try_apply(|v| adder(tu)(&offset, v, Some(&tz))),
                        Err(_) => unreachable!(),
                    },
                },
                _ => ca.0.try_apply(|v| adder(tu)(&offset, v, NO_TIMEZONE)),
            }?;
            out.cast(&DataType::Datetime(tu, tz))
        }
        dt => polars_bail!(
            ComputeError: "cannot use 'date_offset' on Series of datatype {}", dt,
        ),
    }
}

pub(super) fn combine(s: &[Series], tu: TimeUnit) -> PolarsResult<Series> {
    let date = &s[0];
    let time = &s[1];

    let date = date.cast(&DataType::Date)?;
    let datetime = date.cast(&DataType::Datetime(tu, None)).unwrap();

    let duration = time.cast(&DataType::Duration(tu))?;
    Ok(datetime + duration)
}
