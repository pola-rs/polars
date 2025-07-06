#[cfg(feature = "timezones")]
use polars_core::prelude::time_zone::parse_time_zone;
use polars_core::prelude::*;
use polars_time::{
    ClosedWindow, Duration, datetime_range_impl_start_end_interval,
    datetime_range_impl_start_end_samples, datetime_range_impl_start_interval_samples,
};

use super::utils::{
    ensure_items_contain_exactly_one_value, temporal_ranges_impl_broadcast,
    temporal_series_to_i64_scalar,
};
use crate::dsl::DateRangeArgs;
use crate::plans::aexpr::function_expr::FieldsMapper;

const CAPACITY_FACTOR: usize = 5;

fn map_to_out_dtype(
    c: &Column,
    dtype_in: &DataType,
    dtype_out: &DataType,
    time_zone: &Option<TimeZone>,
) -> PolarsResult<Column> {
    match (dtype_in, time_zone) {
        // If `start` and `end` are naive, but a time zone was specified,
        // then first localize them
        #[cfg(feature = "timezones")]
        (DataType::Datetime(_, None), Some(tz)) => Ok(polars_ops::prelude::replace_time_zone(
            c.datetime().unwrap(),
            Some(tz),
            &StringChunked::from_iter(std::iter::once("raise")),
            NonExistent::Raise,
        )?
        .cast(dtype_out)?
        .into_column()),
        _ => c.cast(dtype_out),
    }
}

fn dt_range_start_end_interval(
    start: &Column,
    end: &Column,
    interval: Duration,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> PolarsResult<Column> {
    let dtype_in = start.dtype();
    ensure_items_contain_exactly_one_value(&[&start, &end], &["start", "end"])?;

    let name = start.name();
    let start = temporal_series_to_i64_scalar(&start)
        .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?;
    let end = temporal_series_to_i64_scalar(&end)
        .ok_or_else(|| polars_err!(ComputeError: "end is an out-of-range time."))?;

    let tz = match time_zone {
        #[cfg(feature = "timezones")]
        Some(tz) => Some(parse_time_zone(tz)?),
        _ => None,
    };
    let result = datetime_range_impl_start_end_interval(
        name.clone(),
        start,
        end,
        interval,
        closed,
        time_unit,
        tz.as_ref(),
    )?;
    Ok(result.cast(&dtype_out).unwrap().into_column())
}

fn dt_ranges_start_end_interval(
    start: &Column,
    end: &Column,
    interval: Duration,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> PolarsResult<Column> {
    let dtype_in = start.dtype();
    ensure_items_contain_exactly_one_value(&[&start, &end], &["start", "end"])?;

    let dtype_out = get_dtype_out(
        &start,
        dtype_in,
        time_unit,
        &time_zone,
        interval.nanoseconds() % 1_000 != 0,
    )?;

    let start = map_to_out_dtype(&start, dtype_in, &dtype_out, &time_zone)?;
    let end = map_to_out_dtype(&end, dtype_in, &dtype_out, &time_zone)?;

    let name = start.name();
    let start = temporal_series_to_i64_scalar(&start)
        .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?;
    let end = temporal_series_to_i64_scalar(&end)
        .ok_or_else(|| polars_err!(ComputeError: "end is an out-of-range time."))?;

    let result = match dtype_out {
        DataType::Datetime(tu, ref tz) => {
            let tz = match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => Some(parse_time_zone(tz)?),
                _ => None,
            };
            datetime_range_impl_start_end_interval(
                name.clone(),
                start,
                end,
                interval,
                closed,
                tu,
                tz.as_ref(),
            )?
        },
        _ => unimplemented!(),
    };
    Ok(result.cast(&dtype_out).unwrap().into_column())
}

fn dt_range_start_end_samples(
    start: &Column,
    end: &Column,
    num_samples: &Column,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> PolarsResult<Column> {
    let dtype_in = start.dtype();
    ensure_items_contain_exactly_one_value(&[&start, &end], &["start", "end"])?;
    let num_samples = num_samples.get(0).unwrap().extract::<u64>().unwrap();

    let dtype_out = get_dtype_out(&start, dtype_in, time_unit, &time_zone, false)?;
    let start = map_to_out_dtype(&start, dtype_in, &dtype_out, &time_zone)?;
    let end = map_to_out_dtype(&end, dtype_in, &dtype_out, &time_zone)?;

    let name = start.name();
    let start = temporal_series_to_i64_scalar(&start)
        .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?;
    let end = temporal_series_to_i64_scalar(&end)
        .ok_or_else(|| polars_err!(ComputeError: "end is an out-of-range time."))?;

    let result = match dtype_out {
        DataType::Datetime(tu, ref tz) => {
            let tz = match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => Some(parse_time_zone(tz)?),
                _ => None,
            };
            datetime_range_impl_start_end_samples(
                name.clone(),
                start,
                end,
                num_samples,
                closed,
                tu,
                tz.as_ref(),
            )?
        },
        _ => unimplemented!(),
    };
    Ok(result.cast(&dtype_out).unwrap().into_column())
}

fn dt_range_start_interval_samples(
    start: &Column,
    interval: Duration,
    num_samples: &Column,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> PolarsResult<Column> {
    let dtype_in = start.dtype();
    ensure_items_contain_exactly_one_value(&[&start], &["start"])?;

    let dtype_out = get_dtype_out(
        &start,
        dtype_in,
        time_unit,
        &time_zone,
        interval.nanoseconds() % 1_000 != 0,
    )?;

    let start = map_to_out_dtype(&start, dtype_in, &dtype_out, &time_zone)?;

    let name = start.name();
    let start = temporal_series_to_i64_scalar(&start)
        .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?;
    let num_samples = num_samples.get(0).unwrap().extract::<u64>().unwrap();
    let result = match dtype_out {
        DataType::Datetime(tu, ref tz) => {
            let tz = match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => Some(parse_time_zone(tz)?),
                _ => None,
            };
            datetime_range_impl_start_interval_samples(
                name.clone(),
                start,
                interval,
                num_samples,
                closed,
                tu,
                tz.as_ref(),
            )?
        },
        _ => unimplemented!(),
    };
    Ok(result.cast(&dtype_out).unwrap().into_column())
}

pub(super) fn date_range(
    s: &[Column],
    interval: Option<Duration>,
    closed: ClosedWindow,
    arg_type: DateRangeArgs,
) -> PolarsResult<Column> {
    let dt_type = DataType::Datetime(TimeUnit::Milliseconds, None);
    match arg_type {
        DateRangeArgs::StartEndInterval => dt_range_start_end_interval(
            &s[0].cast(&dt_type)?,
            &s[1].cast(&dt_type)?,
            interval.unwrap(),
            closed,
            None,
            None,
        ),
        DateRangeArgs::StartEndSamples => dt_range_start_end_samples(
            &s[0].cast(&dt_type)?,
            &s[1].cast(&dt_type)?,
            &s[2],
            closed,
            None,
            None,
        ),
        DateRangeArgs::StartIntervalSamples => dt_range_start_interval_samples(
            &s[0].cast(&dt_type)?,
            interval.unwrap(),
            &s[1],
            closed,
            None,
            None,
        ),
        // We negate the interval, start at the end, and then reverse.
        DateRangeArgs::EndIntervalSamples => dt_range_start_interval_samples(
            &s[0].cast(&dt_type)?,
            -interval.unwrap(),
            &s[1],
            closed,
            None,
            None,
        ),
    }
    .map(|c| c.cast(&DataType::Date))?
}

pub(super) fn datetime_range(
    s: &[Column],
    interval: Option<Duration>,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
    arg_type: DateRangeArgs,
) -> PolarsResult<Column> {
    match arg_type {
        DateRangeArgs::StartEndInterval => dt_range_start_end_interval(
            &s[0],
            &s[1],
            interval.unwrap(),
            closed,
            time_unit,
            time_zone.clone(),
        ),
        DateRangeArgs::StartEndSamples => {
            dt_range_start_end_samples(&s[0], &s[1], &s[2], closed, time_unit, time_zone.clone())
        },
        DateRangeArgs::StartIntervalSamples => dt_range_start_interval_samples(
            &s[0],
            interval.unwrap(),
            &s[1],
            closed,
            time_unit,
            time_zone.clone(),
        ),
        // We negate the interval, start at the end, and then reverse.
        DateRangeArgs::EndIntervalSamples => dt_range_start_interval_samples(
            &s[0],
            -interval.unwrap(),
            &s[1],
            closed,
            time_unit,
            time_zone.clone(),
        )
        .map(|c| c.reverse()),
    }
}

pub(super) fn datetime_ranges(
    s: &[Column],
    interval: Option<Duration>,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
    _arg_type: DateRangeArgs,
) -> PolarsResult<Column> {
    match arg_type {
        DateRangeArgs::StartEndInterval => dt_ranges_start_end_interval(
            &s[0],
            &s[1],
            interval.unwrap(),
            closed,
            time_unit,
            time_zone.clone(),
        ),
        DateRangeArgs::StartEndSamples => {
            dt_range_start_end_samples(&s[0], &s[1], &s[2], closed, time_unit, time_zone.clone())
        },
        DateRangeArgs::StartIntervalSamples => dt_range_start_interval_samples(
            &s[0],
            interval.unwrap(),
            &s[1],
            closed,
            time_unit,
            time_zone.clone(),
        ),
        // We negate the interval, start at the end, and then reverse.
        DateRangeArgs::EndIntervalSamples => dt_range_start_interval_samples(
            &s[0],
            -interval.unwrap(),
            &s[1],
            closed,
            time_unit,
            time_zone.clone(),
        )
        .map(|c| c.reverse()),
    }

    let dtype_in = s[0].dtype();
    let mut start = s[0].clone();
    let mut end = s[1].clone();
    let interval = interval.unwrap();

    let dtype_out = get_dtype_out(
        &start,
        dtype_in,
        time_unit,
        &time_zone,
        interval.nanoseconds() % 1_000 != 0,
    )?;

    let start = map_to_out_dtype(&start, dtype_in, &dtype_out, &time_zone)?;
    let end = map_to_out_dtype(&end, dtype_in, &dtype_out, &time_zone)?;

    let start = start.i64().unwrap();
    let end = end.i64().unwrap();

    let out = match dtype_out {
        DataType::Datetime(tu, ref tz) => {
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
                let rng = datetime_range_impl_start_end_interval(
                    PlSmallStr::EMPTY,
                    start,
                    end,
                    interval,
                    closed,
                    tu,
                    tz.as_ref(),
                )?;
                builder.append_slice(rng.physical().cont_slice().unwrap());
                Ok(())
            };

            temporal_ranges_impl_broadcast(start, end, range_impl, &mut builder)?
        },
        _ => unimplemented!(),
    };

    let to_type = DataType::List(Box::new(dtype_out));
    out.cast(&to_type)
}

impl FieldsMapper<'_> {
    pub(super) fn map_to_datetime_range_dtype(
        &self,
        time_unit: Option<&TimeUnit>,
        time_zone: Option<&TimeZone>,
    ) -> PolarsResult<DataType> {
        let data_dtype = self.map_to_supertype()?.dtype;

        let (data_tu, data_tz) = if let DataType::Datetime(tu, tz) = data_dtype {
            (tu, tz)
        } else {
            (TimeUnit::Microseconds, None)
        };

        let tu = match time_unit {
            Some(tu) => *tu,
            None => data_tu,
        };

        let tz = time_zone.cloned().or(data_tz);

        Ok(DataType::Datetime(tu, tz))
    }
}
