#[cfg(feature = "timezones")]
use polars_core::prelude::time_zone::parse_time_zone;
use polars_core::prelude::*;
#[cfg(feature = "dtype-date")]
use polars_plan::dsl::DateRangeArgs;
use polars_time::{
    ClosedWindow, Duration, datetime_range_impl_start_end_interval,
    datetime_range_impl_start_end_samples, datetime_range_impl_start_interval_samples,
};

use super::utils::{
    ensure_items_contain_exactly_one_value, temporal_ranges_impl_broadcast_2args,
    temporal_ranges_impl_broadcast_3args, temporal_series_to_i64_scalar,
};

const CAPACITY_FACTOR: usize = 5;

// Swap left / right closure in the event of an interval inversion.
fn swap_closed_lr(closed: ClosedWindow) -> ClosedWindow {
    match closed {
        ClosedWindow::Left => ClosedWindow::Right,
        ClosedWindow::Right => ClosedWindow::Left,
        _ => closed,
    }
}

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
        DateRangeArgs::StartEndSamples => {
            dt_range_start_end_samples(&s[0].cast(&dt_type)?, &s[1].cast(&dt_type)?, &s[2], closed)
        },
        DateRangeArgs::StartIntervalSamples => {
            dt_range_start_interval_samples(&s[0].cast(&dt_type)?, interval.unwrap(), &s[1], closed)
        },
        // We negate the interval, start at the end, and then reverse.,
        DateRangeArgs::EndIntervalSamples => dt_range_start_interval_samples(
            &s[0].cast(&dt_type)?,
            -interval.unwrap(),
            &s[1],
            swap_closed_lr(closed),
        )
        .map(|c| c.reverse()),
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
        DateRangeArgs::StartEndSamples => {
            dt_ranges_start_end_samples(&s[0].cast(&dt_type)?, &s[1].cast(&dt_type)?, &s[2], closed)
        },
        DateRangeArgs::StartIntervalSamples => dt_ranges_start_interval_samples(
            &s[0].cast(&dt_type)?,
            interval.unwrap(),
            &s[1],
            closed,
        ),
        DateRangeArgs::EndIntervalSamples => {
            dt_ranges_end_interval_samples(&s[0].cast(&dt_type)?, interval.unwrap(), &s[1], closed)
        },
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
        DateRangeArgs::StartEndSamples => dt_range_start_end_samples(&s[0], &s[1], &s[2], closed),
        DateRangeArgs::StartIntervalSamples => {
            dt_range_start_interval_samples(&s[0], interval.unwrap(), &s[1], closed)
        },
        DateRangeArgs::EndIntervalSamples => dt_range_start_interval_samples(
            &s[0],
            -interval.unwrap(),
            &s[1],
            swap_closed_lr(closed),
        )
        .map(|c| c.reverse()),
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
        DateRangeArgs::StartEndSamples => dt_ranges_start_end_samples(&s[0], &s[1], &s[2], closed),
        DateRangeArgs::StartIntervalSamples => {
            dt_ranges_start_interval_samples(&s[0], interval.unwrap(), &s[1], closed)
        },
        DateRangeArgs::EndIntervalSamples => {
            dt_ranges_end_interval_samples(&s[0], interval.unwrap(), &s[1], closed)
        },
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
        let result = datetime_range_impl_start_end_interval(
            name.clone(),
            start,
            end,
            interval,
            closed,
            *tu,
            tz.as_ref(),
        )?;
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
            let rng = datetime_range_impl_start_end_interval(
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

        temporal_ranges_impl_broadcast_2args(start, end, range_impl, &mut builder)?
    } else {
        polars_bail!(ComputeError: "expected Datetime input, got {:?}", dtype);
    };

    let to_type = DataType::List(Box::new(dtype.clone()));
    out.cast(&to_type)
}

fn dt_range_start_end_samples(
    start: &Column,
    end: &Column,
    num_samples: &Column,
    closed: ClosedWindow,
) -> PolarsResult<Column> {
    ensure_items_contain_exactly_one_value(&[start, end], &["start", "end"])?;
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
        let num_samples = num_samples.get(0).unwrap().extract::<i64>().unwrap();
        let result = datetime_range_impl_start_end_samples(
            name.clone(),
            start,
            end,
            num_samples,
            closed,
            *tu,
            tz.as_ref(),
        )?;
        Ok(result.into_column())
    } else {
        polars_bail!(ComputeError: "expected Datetime input, got {:?}", dtype);
    }
}

fn dt_ranges_start_end_samples(
    start: &Column,
    end: &Column,
    num_samples: &Column,
    closed: ClosedWindow,
) -> PolarsResult<Column> {
    let dtype = start.dtype();
    let start = start.to_physical_repr();
    let start = start.i64()?;
    let end = end.to_physical_repr();
    let end = end.i64()?;
    let num_samples = num_samples.i64()?;

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
        let range_impl =
            |start, end, num_samples, builder: &mut ListPrimitiveChunkedBuilder<Int64Type>| {
                let rng = datetime_range_impl_start_end_samples(
                    PlSmallStr::EMPTY,
                    start,
                    end,
                    num_samples,
                    closed,
                    *tu,
                    tz.as_ref(),
                )?;
                builder.append_slice(rng.physical().cont_slice().unwrap());
                Ok(())
            };

        temporal_ranges_impl_broadcast_3args(start, end, num_samples, range_impl, &mut builder)?
    } else {
        polars_bail!(ComputeError: "expected Datetime input, got {:?}", dtype);
    };

    let to_type = DataType::List(Box::new(dtype.clone()));
    out.cast(&to_type)
}

fn dt_range_start_interval_samples(
    start: &Column,
    interval: Duration,
    num_samples: &Column,
    closed: ClosedWindow,
) -> PolarsResult<Column> {
    ensure_items_contain_exactly_one_value(&[start, num_samples], &["start", "num_samples"])?;
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
        let num_samples = num_samples.get(0).unwrap().extract::<i64>().unwrap();
        let result = datetime_range_impl_start_interval_samples(
            name.clone(),
            start,
            interval,
            num_samples,
            closed,
            *tu,
            tz.as_ref(),
        )?;
        Ok(result.into_column())
    } else {
        polars_bail!(ComputeError: "Expected Datetime input, got {:?}", dtype);
    }
}

fn dt_ranges_start_interval_samples(
    start: &Column,
    interval: Duration,
    num_samples: &Column,
    closed: ClosedWindow,
) -> PolarsResult<Column> {
    let dtype = start.dtype();
    let start = start.to_physical_repr();
    let start = start.i64()?;
    let num_samples = num_samples.i64()?;

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
        let range_impl =
            |start, num_samples, builder: &mut ListPrimitiveChunkedBuilder<Int64Type>| {
                let rng = datetime_range_impl_start_interval_samples(
                    PlSmallStr::EMPTY,
                    start,
                    interval,
                    num_samples,
                    closed,
                    *tu,
                    tz.as_ref(),
                )?;
                builder.append_slice(rng.physical().cont_slice().unwrap());
                Ok(())
            };

        temporal_ranges_impl_broadcast_2args(start, num_samples, range_impl, &mut builder)?
    } else {
        polars_bail!(ComputeError: "expected Datetime input, got {:?}", dtype);
    };

    let to_type = DataType::List(Box::new(dtype.clone()));
    out.cast(&to_type)
}

fn dt_ranges_end_interval_samples(
    end: &Column,
    interval: Duration,
    num_samples: &Column,
    closed: ClosedWindow,
) -> PolarsResult<Column> {
    let dtype = end.dtype();
    let end = end.to_physical_repr();
    let end = end.i64()?;
    let num_samples = num_samples.i64()?;

    let out = if let DataType::Datetime(tu, time_zone) = dtype {
        let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
            end.name().clone(),
            end.len(),
            end.len() * CAPACITY_FACTOR,
            DataType::Int64,
        );

        let tz = match time_zone {
            #[cfg(feature = "timezones")]
            Some(tz) => Some(parse_time_zone(tz)?),
            _ => None,
        };
        let range_impl =
            |end, num_samples, builder: &mut ListPrimitiveChunkedBuilder<Int64Type>| {
                let rng = datetime_range_impl_start_interval_samples(
                    PlSmallStr::EMPTY,
                    end,
                    -interval,
                    num_samples,
                    swap_closed_lr(closed),
                    *tu,
                    tz.as_ref(),
                )?;
                builder.append_slice(rng.physical().reverse().cont_slice().unwrap());
                Ok(())
            };

        temporal_ranges_impl_broadcast_2args(end, num_samples, range_impl, &mut builder)?
    } else {
        polars_bail!(ComputeError: "expected Datetime input, got {:?}", dtype);
    };

    let to_type = DataType::List(Box::new(dtype.clone()));
    out.cast(&to_type)
}
