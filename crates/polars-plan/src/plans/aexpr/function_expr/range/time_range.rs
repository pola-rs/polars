use polars_core::prelude::*;
use polars_time::{ClosedWindow, Duration, time_range_impl};

use super::utils::{
    ensure_items_contain_exactly_one_value, temporal_ranges_impl_broadcast_2args,
    temporal_series_to_i64_scalar,
};

const CAPACITY_FACTOR: usize = 5;

pub(super) fn time_range(
    s: &[Column],
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<Column> {
    let start = &s[0];
    let end = &s[1];
    let name = start.name();

    ensure_items_contain_exactly_one_value(&[start, end], &["start", "end"])?;

    let dtype = DataType::Time;
    let start = temporal_series_to_i64_scalar(&start.cast(&dtype)?)
        .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?;
    let end = temporal_series_to_i64_scalar(&end.cast(&dtype)?)
        .ok_or_else(|| polars_err!(ComputeError: "end is an out-of-range time."))?;

    let out = time_range_impl(name.clone(), start, end, interval, closed)?;
    Ok(out.cast(&dtype).unwrap().into_column())
}

pub(super) fn time_ranges(
    s: &[Column],
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<Column> {
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
        start.name().clone(),
        len,
        len * CAPACITY_FACTOR,
        DataType::Int64,
    );

    let range_impl = |start, end, builder: &mut ListPrimitiveChunkedBuilder<Int64Type>| {
        let rng = time_range_impl(PlSmallStr::EMPTY, start, end, interval, closed)?;
        builder.append_slice(rng.physical().cont_slice().unwrap());
        Ok(())
    };

    let out = temporal_ranges_impl_broadcast_2args(start, end, range_impl, &mut builder)?;

    let to_type = DataType::List(Box::new(DataType::Time));
    out.cast(&to_type)
}
