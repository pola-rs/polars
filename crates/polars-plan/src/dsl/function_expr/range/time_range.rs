use polars_core::prelude::*;
use polars_core::series::Series;
use polars_time::{time_range_impl, ClosedWindow, Duration};

use super::utils::{ensure_range_bounds_contain_exactly_one_value, temporal_series_to_i64_scalar};

const CAPACITY_FACTOR: usize = 5;

pub(super) fn time_range(
    s: &[Series],
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<Series> {
    let start = &s[0];
    let end = &s[1];

    ensure_range_bounds_contain_exactly_one_value(start, end)?;

    let dtype = DataType::Time;
    let start = temporal_series_to_i64_scalar(&start.cast(&dtype)?)
        .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?;
    let end = temporal_series_to_i64_scalar(&end.cast(&dtype)?)
        .ok_or_else(|| polars_err!(ComputeError: "end is an out-of-range time."))?;

    let out = time_range_impl("time", start, end, interval, closed)?;
    Ok(out.cast(&dtype).unwrap().into_series())
}

pub(super) fn time_ranges(
    s: &[Series],
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<Series> {
    let start = &s[0];
    let end = &s[1];

    polars_ensure!(
        start.len() == end.len(),
        ComputeError: "`start` and `end` must have the same length",
    );

    let start = time_series_to_i64_ca(start)?;
    let end = time_series_to_i64_ca(end)?;

    let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
        "time_range",
        start.len(),
        start.len() * CAPACITY_FACTOR,
        DataType::Int64,
    );
    for (start, end) in start.as_ref().into_iter().zip(&end) {
        match (start, end) {
            (Some(start), Some(end)) => {
                let rng = time_range_impl("", start, end, interval, closed)?;
                builder.append_slice(rng.cont_slice().unwrap())
            },
            _ => builder.append_null(),
        }
    }
    let list = builder.finish().into_series();

    let to_type = DataType::List(Box::new(DataType::Time));
    list.cast(&to_type)
}
fn time_series_to_i64_ca(s: &Series) -> PolarsResult<ChunkedArray<Int64Type>> {
    let s = s.cast(&DataType::Time)?;
    let s = s.to_physical_repr();
    let result = s.i64().unwrap();
    Ok(result.clone())
}
