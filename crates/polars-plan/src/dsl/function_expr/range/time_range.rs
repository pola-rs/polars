use polars_core::prelude::*;
use polars_time::{time_range_impl, ClosedWindow, Duration};

use super::utils::{
    ensure_range_bounds_contain_exactly_one_value, temporal_ranges_impl_broadcast,
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
    let name = start.name();

    ensure_range_bounds_contain_exactly_one_value(start, end)?;

    let dtype = DataType::Time;
    let start = temporal_series_to_i64_scalar(&start.cast(&dtype)?)
        .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?;
    let end = temporal_series_to_i64_scalar(&end.cast(&dtype)?)
        .ok_or_else(|| polars_err!(ComputeError: "end is an out-of-range time."))?;

    let out = time_range_impl(name, start, end, interval, closed)?;
    Ok(out.cast(&dtype).unwrap().into_series())
}

pub(super) fn time_ranges(
    s: &[Series],
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<Series> {
    let start = &s[0];
    let end = &s[1];

    let start = start.cast(&DataType::Time)?;
    let end = end.cast(&DataType::Time)?;

    let start_phys = start.to_physical_repr();
    let end_phys = end.to_physical_repr();
    let start = start_phys.i64().unwrap();
    let end = end_phys.i64().unwrap();

    let len = std::cmp::max(start.len(), end.len());
    let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
        start.name(),
        len,
        len * CAPACITY_FACTOR,
        DataType::Int64,
    );

    let range_impl = |start, end, builder: &mut ListPrimitiveChunkedBuilder<Int64Type>| {
        let rng = time_range_impl("", start, end, interval, closed)?;
        builder.append_slice(rng.cont_slice().unwrap());
        Ok(())
    };

    let out = temporal_ranges_impl_broadcast(start, end, range_impl, &mut builder)?;

    let to_type = DataType::List(Box::new(DataType::Time));
    out.cast(&to_type)
}
