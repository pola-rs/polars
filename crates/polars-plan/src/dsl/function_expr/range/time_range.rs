use polars_core::prelude::*;
use polars_core::series::Series;
use polars_time::{time_range_impl, ClosedWindow, Duration};

use super::utils::{
    broadcast_scalar_inputs, ensure_range_bounds_contain_exactly_one_value,
    temporal_series_to_i64_scalar,
};

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

    let mut start = start.cast(&DataType::Time)?;
    let mut end = end.cast(&DataType::Time)?;

    (start, end) = broadcast_scalar_inputs(start, end)?;

    let start = start.to_physical_repr();
    let start = start.i64().unwrap();
    let end = end.to_physical_repr();
    let end = end.i64().unwrap();

    let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
        "time_range",
        start.len(),
        start.len() * CAPACITY_FACTOR,
        DataType::Int64,
    );
    for (start, end) in start.into_iter().zip(end) {
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
