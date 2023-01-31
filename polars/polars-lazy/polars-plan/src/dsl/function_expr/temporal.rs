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
        DataType::Datetime(tu, _) => {
            // drop series, so that we might modify in place
            let mut ca = {
                let me = std::mem::ManuallyDrop::new(s);
                me.datetime().unwrap().clone()
            };

            let adder = match tu {
                TimeUnit::Nanoseconds => Duration::add_ns,
                TimeUnit::Microseconds => Duration::add_us,
                TimeUnit::Milliseconds => Duration::add_ms,
            };
            ca.0.apply_mut(|v| adder(&offset, v));
            Ok(ca.into_series())
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
