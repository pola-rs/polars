#[cfg(feature = "timezones")]
use polars_core::prelude::time_zone::parse_time_zone;
use polars_core::prelude::*;
use polars_time::{ClosedWindow, Duration, datetime_range_impl};

use super::utils::{
    ensure_range_bounds_contain_exactly_one_value, temporal_ranges_impl_broadcast,
    temporal_series_to_i64_scalar,
};

const CAPACITY_FACTOR: usize = 5;

pub(super) fn datetime_range(
    s: &[Column],
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<Column> {
    let start = s[0].clone();
    let end = s[1].clone();
    let dtype = start.dtype();

    ensure_range_bounds_contain_exactly_one_value(&start, &end)?;

    let name = start.name();
    let start = temporal_series_to_i64_scalar(&start)
        .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?;
    let end = temporal_series_to_i64_scalar(&end)
        .ok_or_else(|| polars_err!(ComputeError: "end is an out-of-range time."))?;

    let result = match dtype {
        DataType::Datetime(tu, tz) => {
            let tz = match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => Some(parse_time_zone(tz)?),
                _ => None,
            };
            datetime_range_impl(name.clone(), start, end, interval, closed, *tu, tz.as_ref())?
        },
        _ => polars_bail!(ComputeError: "expected 'Datetime', got '{}'", dtype),
    };
    Ok(result.cast(dtype).unwrap().into_column())
}

pub(super) fn datetime_ranges(
    s: &[Column],
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<Column> {
    let start = s[0].clone();
    let end = s[1].clone();
    let dtype = start.dtype();

    let start = start.cast(&DataType::Int64)?;
    let start = start.i64().unwrap();
    let end = end.cast(&DataType::Int64)?;
    let end = end.i64().unwrap();

    let out = match dtype {
        DataType::Datetime(tu, tz) => {
            let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
                start.name().clone(),
                start.len(),
                start.len() * CAPACITY_FACTOR,
                DataType::Int64,
            );

            let tz = match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => Some(parse_time_zone(tz)?),
                _ => None,
            };
            let range_impl = |start, end, builder: &mut ListPrimitiveChunkedBuilder<Int64Type>| {
                let rng = datetime_range_impl(
                    PlSmallStr::EMPTY,
                    start,
                    end,
                    interval,
                    closed,
                    *tu,
                    tz.as_ref(),
                )?;
                builder.append_slice(rng.physical().cont_slice().unwrap());
                Ok(())
            };

            temporal_ranges_impl_broadcast(start, end, range_impl, &mut builder)?
        },
        _ => polars_bail!(ComputeError: "expected 'Datetime', got '{}'", dtype),
    };

    let to_type = DataType::List(Box::new(dtype.clone()));
    out.cast(&to_type)
}
