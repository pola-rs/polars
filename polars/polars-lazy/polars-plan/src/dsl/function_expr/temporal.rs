#[cfg(feature = "date_offset")]
use polars_time::prelude::*;

use super::*;

#[cfg(feature = "date_offset")]
pub(super) fn date_offset(s: &[Series]) -> PolarsResult<Series> {
    let date = &s[0];
    let offset = &s[1];

    match date.dtype().clone() {
        DataType::Date => {
            let date = date
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap();
            date_offset(&[date, offset.clone()]).and_then(|s| s.cast(&DataType::Date))
        }
        DataType::Datetime(tu, _) => {
            let adder = match tu {
                TimeUnit::Nanoseconds => Duration::add_ns,
                TimeUnit::Microseconds => Duration::add_us,
                TimeUnit::Milliseconds => Duration::add_ms,
            };
            let offset: Vec<Duration> = offset
                .utf8()
                .unwrap()
                .into_iter()
                .map(|opt_offset| match opt_offset {
                    Some(opt_offset) => Duration::parse(&opt_offset),
                    _ => Duration::parse("0"),
                })
                .collect();
            let series = if offset.len() == 1 {
                let mut ca = std::mem::ManuallyDrop::new(date)
                    .datetime()
                    .unwrap()
                    .clone();
                let opt_offset = offset.get(0);
                if opt_offset.is_some() {
                    let offset = opt_offset.unwrap();
                    ca.0.apply_mut(|date| adder(&offset, date))
                }
                ca.into_series()
            } else {
                let values: Vec<AnyValue> = date
                    .datetime()
                    .unwrap()
                    .into_iter()
                    .zip(offset.into_iter())
                    .map(|(opt_date, opt_duration)| match (opt_date, opt_duration) {
                        (Some(date), duration) => AnyValue::Int64(adder(&duration, date)),
                        _ => AnyValue::Null,
                    })
                    .collect();
                Series::from_any_values_and_dtype("", &values, date.dtype())?
            };
            Ok(series)
        }
        dt => Err(PolarsError::ComputeError(
            format!("cannot use 'date_offset' on Series of dtype: {dt:?}").into(),
        )),
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
