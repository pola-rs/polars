#[cfg(feature = "timezones")]
use polars_core::prelude::time_zone::parse_time_zone;
use polars_core::prelude::*;
#[cfg(feature = "dtype-date")]
use polars_plan::dsl::DateRangeArgs;
use polars_time::{ClosedWindow, Duration, datetime_range_impl};

use super::utils::{
    ensure_items_contain_exactly_one_value, temporal_ranges_impl_broadcast,
    temporal_series_to_i64_scalar,
};

const CAPACITY_FACTOR: usize = 5;

#[cfg(feature = "dtype-date")]
pub(super) fn date_range(
    s: &[Column],
    interval: Option<Duration>,
    closed: ClosedWindow,
    arg_type: DateRangeArgs,
) -> PolarsResult<Column> {
    let dt_type = DataType::Datetime(TimeUnit::Microseconds, None);
    match arg_type {
        DateRangeArgs::StartEndInterval => dt_range_start_end_interval(
            &s[0].cast(&dt_type)?,
            &s[1].cast(&dt_type)?,
            interval.unwrap(),
            closed,
        ),
        DateRangeArgs::StartEndSamples => todo!(),
        DateRangeArgs::StartIntervalSamples => todo!(),
        DateRangeArgs::EndIntervalSamples => todo!(),
    }
    .map(|c| c.cast(&DataType::Date))?
}

#[cfg(feature = "dtype-date")]
pub(super) fn date_ranges(
    s: &[Column],
    interval: Option<Duration>,
    closed: ClosedWindow,
    arg_type: DateRangeArgs,
) -> PolarsResult<Column> {
    let dt_type = DataType::Datetime(TimeUnit::Microseconds, None);
    match arg_type {
        DateRangeArgs::StartEndInterval => dt_ranges_start_end_interval(
            &s[0].cast(&dt_type)?,
            &s[1].cast(&dt_type)?,
            interval.unwrap(),
            closed,
        ),
        DateRangeArgs::StartEndSamples => todo!(),
        DateRangeArgs::StartIntervalSamples => todo!(),
        DateRangeArgs::EndIntervalSamples => todo!(),
    }
    .map(|c| c.cast(&DataType::List(Box::new(DataType::Date))))?
}

#[cfg(feature = "dtype-datetime")]
pub(super) fn datetime_range(
    s: &[Column],
    interval: Option<Duration>,
    closed: ClosedWindow,
    arg_type: DateRangeArgs,
) -> PolarsResult<Column> {
    match arg_type {
        DateRangeArgs::StartEndInterval => {
            dt_range_start_end_interval(&s[0], &s[1], interval.unwrap(), closed)
        },
        DateRangeArgs::StartEndSamples => todo!(),
        DateRangeArgs::StartIntervalSamples => todo!(),
        DateRangeArgs::EndIntervalSamples => todo!(),
    }
}

#[cfg(feature = "dtype-datetime")]
pub(super) fn datetime_ranges(
    s: &[Column],
    interval: Option<Duration>,
    closed: ClosedWindow,
    arg_type: DateRangeArgs,
) -> PolarsResult<Column> {
    match arg_type {
        DateRangeArgs::StartEndInterval => {
            dt_ranges_start_end_interval(&s[0], &s[1], interval.unwrap(), closed)
        },
        DateRangeArgs::StartEndSamples => todo!(),
        DateRangeArgs::StartIntervalSamples => todo!(),
        DateRangeArgs::EndIntervalSamples => todo!(),
    }
}

/// Datetime range given start, end, and interval.
fn dt_range_start_end_interval(
    start: &Column,
    end: &Column,
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<Column> {
    ensure_items_contain_exactly_one_value(&[start, end], &["start", "end"])?;
    let dtype = start.dtype();

    if let DataType::Datetime(tu, time_zone) = dtype {
        let tz = match time_zone {
            #[cfg(feature = "timezones")]
            Some(tz) => Some(parse_time_zone(tz)?),
            _ => None,
        };
        let name = start.name();
        let start = temporal_series_to_i64_scalar(start)
            .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?;
        let end = temporal_series_to_i64_scalar(end)
            .ok_or_else(|| polars_err!(ComputeError: "end is an out-of-range time."))?;
        let result =
            datetime_range_impl(name.clone(), start, end, interval, closed, *tu, tz.as_ref())?;
        Ok(result.into_column())
    } else {
        polars_bail!(ComputeError: "expected Datetime input, got {:?}", dtype);
    }
}

fn dt_ranges_start_end_interval(
    start: &Column,
    end: &Column,
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<Column> {
    let dtype = start.dtype();

    let start = start.to_physical_repr();
    let start = start.i64()?;
    let end = end.to_physical_repr();
    let end = end.i64()?;

    let out = if let DataType::Datetime(tu, time_zone) = dtype {
        let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
            start.name().clone(),
            start.len(),
            start.len() * CAPACITY_FACTOR,
            DataType::Int64,
        );

        let tz = match time_zone {
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
    } else {
        polars_bail!(ComputeError: "expected Datetime input, got {:?}", dtype);
    };

    let to_type = DataType::List(Box::new(dtype.clone()));
    out.cast(&to_type)
}
