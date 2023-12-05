use std::iter::zip;

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

    let start = start.cast(&DataType::Time)?;
    let end = end.cast(&DataType::Time)?;

    let start_phys = start.to_physical_repr();
    let end_phys = end.to_physical_repr();
    let start = start_phys.i64().unwrap();
    let end = end_phys.i64().unwrap();

    let len = std::cmp::max(start.len(), end.len());
    let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
        "time_range",
        len,
        len * CAPACITY_FACTOR,
        DataType::Int64,
    );

    let range_impl = |start, end| time_range_impl("", start, end, interval, closed);

    match (start.len(), end.len()) {
        (len_start, len_end) if len_start == len_end => {
            ranges_double_impl(&mut builder, start, end, range_impl)?;
        },
        (1, len_end) => {
            let start_scalar = unsafe { start.get_unchecked(0) };
            match start_scalar {
                Some(start) => {
                    let range_impl = |end| range_impl(start, end);
                    for end_scalar in end {
                        match end_scalar {
                            Some(end) => {
                                let rng = range_impl(end)?;
                                builder.append_slice(rng)
                            },
                            None => builder.append_null(),
                        }
                    }
                },
                None => {
                    for _ in 0..len_end {
                        builder.append_null()
                    }
                },
            }
        },
        (len_start, 1) => {
            let end_scalar = unsafe { end.get_unchecked(0) };
            match end_scalar {
                Some(end) => {
                    let range_impl = |start| range_impl(start, end);
                    for start_scalar in start {
                        match start_scalar {
                            Some(start) => {
                                let rng = range_impl(start)?;
                                builder.append_slice(rng)
                            },
                            None => builder.append_null(),
                        }
                    }
                },
                None => {
                    for _ in 0..len_start {
                        builder.append_null()
                    }
                },
            }
        },
        (len_start, len_end) => {
            polars_bail!(
                ComputeError:
                "lengths of `start` ({}) and `end` ({}) do not match",
                len_start, len_end
            )
        },
    };
    let out = builder.finish().into_series();

    let to_type = DataType::List(Box::new(DataType::Time));
    out.cast(&to_type)
}

fn ranges_double_impl<T, F>(
    builder: &mut ListPrimitiveChunkedBuilder<T>,
    start: &ChunkedArray<T>,
    end: &ChunkedArray<T>,
    range_impl: F,
) -> PolarsResult<()>
where
    T: PolarsIntegerType,
    F: Fn(T::Native, T::Native) -> PolarsResult<ChunkedArray>,
{
    for (start, end) in zip(start, end) {
        match (start, end) {
            (Some(start), Some(end)) => {
                let rng = range_impl(start, end)?;
                builder.append_slice(rng.cont_slice().unwrap())
            },
            _ => builder.append_null(),
        }
    }
    Ok(())
}
