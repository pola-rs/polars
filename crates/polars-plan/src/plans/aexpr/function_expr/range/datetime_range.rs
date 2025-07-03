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

fn get_dtype_out(
    column: &Column,
    dtype_in: &DataType,
    time_unit: Option<TimeUnit>,
    time_zone: &Option<TimeZone>,
    interval_has_ns: bool,
) -> PolarsResult<DataType> {
    // Note: `start` and `end` have already been cast to their supertype,
    // so only `start`'s dtype needs to be matched against.
    #[allow(unused_mut)] // `dtype` is mutated within a "feature = timezones" block.
    let mut dtype_out = match (dtype_in, time_unit) {
        (DataType::Date, time_unit) => {
            if let Some(tu) = time_unit {
                DataType::Datetime(tu, None)
            } else if interval_has_ns {
                DataType::Datetime(TimeUnit::Nanoseconds, None)
            } else {
                DataType::Datetime(TimeUnit::Microseconds, None)
            }
        },
        // overwrite nothing, keep as-is
        (DataType::Datetime(_, _), None) => column.dtype().clone(),
        // overwrite time unit, keep timezone
        (DataType::Datetime(_, tz), Some(tu)) => DataType::Datetime(tu, tz.clone()),
        (dt, _) => polars_bail!(InvalidOperation: "expected a temporal datatype, got {}", dt),
    };

    // Overwrite time zone, if specified
    #[cfg(feature = "timezones")]
    if let (DataType::Datetime(tu, _), Some(tz)) = (&dtype_out, &time_zone) {
        dtype_out = DataType::Datetime(*tu, Some(tz.clone()));
    };
    Ok(dtype_out)
}

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
        .cast(&dtype_out)?
        .into_column()),
        _ => c.cast(&dtype_out),
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
    let is_date = dtype_in == &DataType::Date;
    let (start, end) = if is_date {
        (
            start.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?,
            end.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?,
        )
    } else {
        (start.clone(), end.clone())
    };
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
    let is_date = dtype_in == &DataType::Date;
    let (start, end) = if is_date {
        (
            start.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?,
            end.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?,
        )
    } else {
        (start.clone(), end.clone())
    };
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
    let is_date = dtype_in == &DataType::Date;
    let start = if is_date {
        start.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?
    } else {
        start.clone()
    };
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
    interval: Duration,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> PolarsResult<Column> {
    let dtype_in = s[0].dtype();
    let mut start = s[0].clone();
    let mut end = s[1].clone();

    let dtype_out = get_dtype_out(
        &start,
        dtype_in,
        time_unit,
        &time_zone,
        interval.nanoseconds() % 1_000 != 0,
    )?;

    if dtype_in == &DataType::Date {
        start = start.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?;
        end = end.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?;
    }

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
                builder.append_slice(rng.cont_slice().unwrap());
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
