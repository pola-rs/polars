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
            // drop series, so that we might modify in place
            let mut ca = {
                let me = std::mem::ManuallyDrop::new(s);
                me.datetime().unwrap().clone()
            };

            fn adder<T: PolarsTimeZone>(
                tu: TimeUnit,
            ) -> fn(&Duration, i64, Option<&T>) -> PolarsResult<i64> {
                match tu {
                    TimeUnit::Nanoseconds => Duration::add_ns,
                    TimeUnit::Microseconds => Duration::add_us,
                    TimeUnit::Milliseconds => Duration::add_ms,
                }
            }

            match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => match tz.parse::<Tz>() {
                    // TODO write `try_apply_mut` and use that instead of `apply_mut`,
                    // then remove `unwrap`.
                    Ok(tz) => {
                        ca.0.apply_mut(|v| adder(tu)(&offset, v, Some(&tz)).unwrap())
                    }
                    Err(_) => match parse_offset(&tz) {
                        Ok(tz) => {
                            ca.0.apply_mut(|v| adder(tu)(&offset, v, Some(&tz)).unwrap())
                        }
                        Err(_) => unreachable!(),
                    },
                },
                _ => {
                    ca.0.apply_mut(|v| adder(tu)(&offset, v, NO_TIMEZONE).unwrap())
                }
            };
            Ok(ca.into_series())
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
