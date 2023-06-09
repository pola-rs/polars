#[cfg(feature = "date_offset")]
use polars_arrow::time_zone::Tz;
use polars_core::utils::arrow::temporal_conversions::SECONDS_IN_DAY;
#[cfg(feature = "date_offset")]
use polars_time::prelude::*;

use super::*;

#[cfg(feature = "date_offset")]
pub(super) fn date_offset(s: Series, offset: Duration) -> PolarsResult<Series> {
    let out = match s.dtype().clone() {
        DataType::Date => {
            let s = s
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap();
            date_offset(s, offset).and_then(|s| s.cast(&DataType::Date))
        }
        DataType::Datetime(tu, tz) => {
            let ca = s.datetime().unwrap();

            fn offset_fn(tu: TimeUnit) -> fn(&Duration, i64, Option<&Tz>) -> PolarsResult<i64> {
                match tu {
                    TimeUnit::Nanoseconds => Duration::add_ns,
                    TimeUnit::Microseconds => Duration::add_us,
                    TimeUnit::Milliseconds => Duration::add_ms,
                }
            }

            let out = match tz {
                #[cfg(feature = "timezones")]
                Some(ref tz) => {
                    let offset_fn = offset_fn(tu);
                    ca.0.try_apply(|v| offset_fn(&offset, v, tz.parse::<Tz>().ok().as_ref()))
                }
                _ => {
                    let offset_fn = offset_fn(tu);
                    ca.0.try_apply(|v| offset_fn(&offset, v, None))
                }
            }?;
            out.cast(&DataType::Datetime(tu, tz))
        }
        dt => polars_bail!(
            ComputeError: "cannot use 'date_offset' on Series of datatype {}", dt,
        ),
    };
    out.map(|mut out| {
        out.set_sorted_flag(s.is_sorted_flag());
        out
    })
}

pub(super) fn combine(s: &[Series], tu: TimeUnit) -> PolarsResult<Series> {
    let date = &s[0];
    let time = &s[1];

    let tz = match date.dtype() {
        DataType::Date => None,
        DataType::Datetime(_, tz) => tz.as_ref(),
        _dtype => {
            polars_bail!(ComputeError: format!("expected Date or Datetime, got {}", _dtype))
        }
    };

    let date = date.cast(&DataType::Date)?;
    let datetime = date.cast(&DataType::Datetime(tu, None)).unwrap();

    let duration = time.cast(&DataType::Duration(tu))?;
    let result_naive = datetime + duration;
    match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => Ok(result_naive
            .datetime()
            .unwrap()
            .replace_time_zone(Some(tz), None)?
            .into()),
        _ => Ok(result_naive),
    }
}

pub(super) fn temporal_range_dispatch(
    s: &[Series],
    name: &str,
    every: Duration,
    closed: ClosedWindow,
    _tz: Option<TimeZone>, // todo: respect _tz: https://github.com/pola-rs/polars/issues/8512
) -> PolarsResult<Series> {
    let start = &s[0];
    let stop = &s[1];

    polars_ensure!(
        start.len() == stop.len(),
        ComputeError: "'start' and 'stop' should have the same length",
    );
    const TO_MS: i64 = SECONDS_IN_DAY * 1000;

    let rng_start = start.to_physical_repr();
    let rng_stop = stop.to_physical_repr();
    let dtype = start.dtype();

    let mut start = rng_start.cast(&DataType::Int64)?;
    let mut stop = rng_stop.cast(&DataType::Int64)?;

    let (tu, tz) = match dtype {
        DataType::Date => {
            start = &start * TO_MS;
            stop = &stop * TO_MS;
            (TimeUnit::Milliseconds, None)
        }
        DataType::Datetime(tu, tz) => (*tu, tz.as_ref()),
        DataType::Time => (TimeUnit::Nanoseconds, None),
        _ => unimplemented!(),
    };
    let start = start.i64().unwrap();
    let stop = stop.i64().unwrap();

    let list = match dtype {
        DataType::Date => {
            let mut builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(
                name,
                start.len(),
                start.len() * 5,
                DataType::Int32,
            );
            for (start, stop) in start.into_iter().zip(stop.into_iter()) {
                match (start, stop) {
                    (Some(start), Some(stop)) => {
                        let rng = date_range_impl("", start, stop, every, closed, tu, tz)?;
                        let rng = rng.cast(&DataType::Date).unwrap();
                        let rng = rng.to_physical_repr();
                        let rng = rng.i32().unwrap();
                        builder.append_slice(rng.cont_slice().unwrap())
                    }
                    _ => builder.append_null(),
                }
            }
            builder.finish().into_series()
        }
        DataType::Datetime(_, _) | DataType::Time => {
            let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
                name,
                start.len(),
                start.len() * 5,
                DataType::Int64,
            );
            for (start, stop) in start.into_iter().zip(stop.into_iter()) {
                match (start, stop) {
                    (Some(start), Some(stop)) => {
                        let rng = date_range_impl("", start, stop, every, closed, tu, tz)?;
                        builder.append_slice(rng.cont_slice().unwrap())
                    }
                    _ => builder.append_null(),
                }
            }
            builder.finish().into_series()
        }
        _ => unimplemented!(),
    };

    let to_type = DataType::List(Box::new(dtype.clone()));
    list.cast(&to_type)
}
