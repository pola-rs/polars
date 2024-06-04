use polars_core::prelude::*;
use polars_core::utils::arrow::temporal_conversions::MILLISECONDS_IN_DAY;
use polars_time::{datetime_range_impl, ClosedWindow, Duration};

use super::utils::{
    ensure_range_bounds_contain_exactly_one_value, temporal_ranges_impl_broadcast,
    temporal_series_to_i64_scalar,
};

const CAPACITY_FACTOR: usize = 5;

pub(super) fn date_range(
    s: &[Series],
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<Series> {
    let start = &s[0];
    let end = &s[1];

    ensure_range_bounds_contain_exactly_one_value(start, end)?;
    let start = start.strict_cast(&DataType::Date)?;
    let end = end.strict_cast(&DataType::Date)?;
    polars_ensure!(
        interval.is_full_days(),
        ComputeError: "`interval` input for `date_range` must consist of full days, got: {interval}"
    );

    let name = start.name();
    let start = temporal_series_to_i64_scalar(&start)
        .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?
        * MILLISECONDS_IN_DAY;
    let end = temporal_series_to_i64_scalar(&end)
        .ok_or_else(|| polars_err!(ComputeError: "end is an out-of-range time."))?
        * MILLISECONDS_IN_DAY;

    let out = datetime_range_impl(
        name,
        start,
        end,
        interval,
        closed,
        TimeUnit::Milliseconds,
        None,
    )?;

    let to_type = DataType::Date;
    out.cast(&to_type)
}

pub(super) fn date_ranges(
    s: &[Series],
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<Series> {
    let start = &s[0];
    let end = &s[1];

    polars_ensure!(
        interval.is_full_days(),
        ComputeError: "`interval` input for `date_ranges` must consist of full days, got: {interval}"
    );

    let start = start.strict_cast(&DataType::Date)?.cast(&DataType::Int64)?;
    let end = end.strict_cast(&DataType::Date)?.cast(&DataType::Int64)?;

    let start = start.i64().unwrap() * MILLISECONDS_IN_DAY;
    let end = end.i64().unwrap() * MILLISECONDS_IN_DAY;

    let mut builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(
        start.name(),
        start.len(),
        start.len() * CAPACITY_FACTOR,
        DataType::Int32,
    );

    let range_impl = |start, end, builder: &mut ListPrimitiveChunkedBuilder<Int32Type>| {
        let rng = datetime_range_impl(
            "",
            start,
            end,
            interval,
            closed,
            TimeUnit::Milliseconds,
            None,
        )?;
        let rng = rng.cast(&DataType::Date).unwrap();
        let rng = rng.to_physical_repr();
        let rng = rng.i32().unwrap();
        builder.append_slice(rng.cont_slice().unwrap());
        Ok(())
    };

    let out = temporal_ranges_impl_broadcast(&start, &end, range_impl, &mut builder)?;

    let to_type = DataType::List(Box::new(DataType::Date));
    out.cast(&to_type)
}
