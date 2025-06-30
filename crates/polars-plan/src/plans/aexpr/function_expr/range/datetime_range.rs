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

    // Note: `start` and `end` have already been cast to their supertype,
    // so only `start`'s dtype needs to be matched against.
    #[allow(unused_mut)] // `dtype` is mutated within a "feature = timezones" block.
    let mut dtype_out = match (dtype_in, time_unit) {
        (DataType::Date, time_unit) => {
            if let Some(tu) = time_unit {
                DataType::Datetime(tu, None)
            } else if interval.nanoseconds() % 1_000 != 0 {
                DataType::Datetime(TimeUnit::Nanoseconds, None)
            } else {
                DataType::Datetime(TimeUnit::Microseconds, None)
            }
        },
        // overwrite nothing, keep as-is
        (DataType::Datetime(_, _), None) => start.dtype().clone(),
        // overwrite time unit, keep timezone
        (DataType::Datetime(_, tz), Some(tu)) => DataType::Datetime(tu, tz.clone()),
        (dt, _) => polars_bail!(InvalidOperation: "expected a temporal datatype, got {}", dt),
    };

    // Overwrite time zone, if specified
    #[cfg(feature = "timezones")]
    if let (DataType::Datetime(tu, _), Some(tz)) = (&dtype_out, &time_zone) {
        dtype_out = DataType::Datetime(*tu, Some(tz.clone()));
    };

    // If `start` and `end` are naive, but a time zone was specified,
    // then first localize them
    let (start, end) = match (dtype_in, time_zone) {
        #[cfg(feature = "timezones")]
        (DataType::Datetime(_, None), Some(tz)) => (
            polars_ops::prelude::replace_time_zone(
                start.datetime().unwrap(),
                Some(&tz),
                &StringChunked::from_iter(std::iter::once("raise")),
                NonExistent::Raise,
            )?
            .cast(&dtype_out)?
            .into_column(),
            polars_ops::prelude::replace_time_zone(
                end.datetime().unwrap(),
                Some(&tz),
                &StringChunked::from_iter(std::iter::once("raise")),
                NonExistent::Raise,
            )?
            .cast(&dtype_out)?
            .into_column(),
        ),
        _ => (start.cast(&dtype_out)?, end.cast(&dtype_out)?),
    };

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

    // Note: `start` and `end` have already been cast to their supertype,
    // so only `start`'s dtype needs to be matched against.
    #[allow(unused_mut)] // `dtype` is mutated within a "feature = timezones" block.
    let mut dtype_out = match (dtype_in, time_unit) {
        (DataType::Date, time_unit) => {
            if let Some(tu) = time_unit {
                DataType::Datetime(tu, None)
            } else {
                DataType::Datetime(TimeUnit::Microseconds, None)
            }
        },
        // overwrite nothing, keep as-is
        (DataType::Datetime(_, _), None) => start.dtype().clone(),
        // overwrite time unit, keep timezone
        (DataType::Datetime(_, tz), Some(tu)) => DataType::Datetime(tu, tz.clone()),
        (dt, _) => polars_bail!(InvalidOperation: "expected a temporal datatype, got {}", dt),
    };

    // Overwrite time zone, if specified
    #[cfg(feature = "timezones")]
    if let (DataType::Datetime(tu, _), Some(tz)) = (&dtype_out, &time_zone) {
        dtype_out = DataType::Datetime(*tu, Some(tz.clone()));
    };

    // If `start` and `end` are naive, but a time zone was specified,
    // then first localize them
    let (start, end) = match (dtype_in, time_zone) {
        #[cfg(feature = "timezones")]
        (DataType::Datetime(_, None), Some(tz)) => (
            polars_ops::prelude::replace_time_zone(
                start.datetime().unwrap(),
                Some(&tz),
                &StringChunked::from_iter(std::iter::once("raise")),
                NonExistent::Raise,
            )?
            .cast(&dtype_out)?
            .into_column(),
            polars_ops::prelude::replace_time_zone(
                end.datetime().unwrap(),
                Some(&tz),
                &StringChunked::from_iter(std::iter::once("raise")),
                NonExistent::Raise,
            )?
            .cast(&dtype_out)?
            .into_column(),
        ),
        _ => (start.cast(&dtype_out)?, end.cast(&dtype_out)?),
    };

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

    #[allow(unused_mut)] // `dtype` is mutated within a "feature = timezones" block.
    let mut dtype_out = match (dtype_in, time_unit) {
        (DataType::Date, time_unit) => {
            if let Some(tu) = time_unit {
                DataType::Datetime(tu, None)
            } else {
                DataType::Datetime(TimeUnit::Microseconds, None)
            }
        },
        // overwrite nothing, keep as-is
        (DataType::Datetime(_, _), None) => dtype_in.clone(),
        // overwrite time unit, keep timezone
        (DataType::Datetime(_, tz), Some(tu)) => DataType::Datetime(tu, tz.clone()),
        (dt, _) => polars_bail!(InvalidOperation: "expected a temporal datatype, got {}", dt),
    };

    // Overwrite time zone, if specified
    #[cfg(feature = "timezones")]
    if let (DataType::Datetime(tu, _), Some(tz)) = (&dtype_out, &time_zone) {
        dtype_out = DataType::Datetime(*tu, Some(tz.clone()));
    };

    // If `start` is naive, but a time zone was specified, then first localize.
    let start = match (dtype_in, time_zone) {
        #[cfg(feature = "timezones")]
        (DataType::Datetime(_, None), Some(tz)) => polars_ops::prelude::replace_time_zone(
            start.datetime().unwrap(),
            Some(&tz),
            &StringChunked::from_iter(std::iter::once("raise")),
            NonExistent::Raise,
        )?
        .cast(&dtype_out)?
        .into_column(),
        _ => start.cast(&dtype_out)?,
    };

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

fn dt_range_end_interval_samples(
    end: &Column,
    interval: Duration,
    num_samples: &Column,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> PolarsResult<Column> {
    let dtype_in = end.dtype();
    let is_date = dtype_in == &DataType::Date;
    let end = if is_date {
        end.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?
    } else {
        end.clone()
    };
    ensure_items_contain_exactly_one_value(&[&end], &["end"])?;

    #[allow(unused_mut)] // `dtype` is mutated within a "feature = timezones" block.
    let mut dtype_out = match (dtype_in, time_unit) {
        (DataType::Date, time_unit) => {
            if let Some(tu) = time_unit {
                DataType::Datetime(tu, None)
            } else {
                DataType::Datetime(TimeUnit::Microseconds, None)
            }
        },
        // overwrite nothing, keep as-is
        (DataType::Datetime(_, _), None) => dtype_in.clone(),
        // overwrite time unit, keep timezone
        (DataType::Datetime(_, tz), Some(tu)) => DataType::Datetime(tu, tz.clone()),
        (dt, _) => polars_bail!(InvalidOperation: "expected a temporal datatype, got {}", dt),
    };

    // Overwrite time zone, if specified
    #[cfg(feature = "timezones")]
    if let (DataType::Datetime(tu, _), Some(tz)) = (&dtype_out, &time_zone) {
        dtype_out = DataType::Datetime(*tu, Some(tz.clone()));
    };

    // If `start` is naive, but a time zone was specified, then first localize.
    let end = match (dtype_in, time_zone) {
        #[cfg(feature = "timezones")]
        (DataType::Datetime(_, None), Some(tz)) => polars_ops::prelude::replace_time_zone(
            end.datetime().unwrap(),
            Some(&tz),
            &StringChunked::from_iter(std::iter::once("raise")),
            NonExistent::Raise,
        )?
        .cast(&dtype_out)?
        .into_column(),
        _ => end.cast(&dtype_out)?,
    };

    let name = end.name();
    let end = temporal_series_to_i64_scalar(&end)
        .ok_or_else(|| polars_err!(ComputeError: "end is an out-of-range time."))?;
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
                end,
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
    // let interval = interval.unwrap();
    let _ = match arg_type {
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
        DateRangeArgs::EndIntervalSamples => dt_range_end_interval_samples(
            &s[0],
            interval.unwrap(),
            &s[1],
            closed,
            time_unit,
            time_zone.clone(),
        ),
    };

    let interval = interval.unwrap();

    let (start, end) = (&s[0], &s[1]);
    ensure_items_contain_exactly_one_value(&[start, end], &["start", "end"])?;
    let dtype_in = start.dtype();

    // Note: `start` and `end` have already been cast to their supertype,
    // so only `start`'s dtype needs to be matched against.
    #[allow(unused_mut)] // `dtype` is mutated within a "feature = timezones" block.
    let mut dtype = match (dtype_in, time_unit) {
        (DataType::Date, time_unit) => {
            if let Some(tu) = time_unit {
                DataType::Datetime(tu, None)
            } else if interval.nanoseconds() % 1_000 != 0 {
                DataType::Datetime(TimeUnit::Nanoseconds, None)
            } else {
                DataType::Datetime(TimeUnit::Microseconds, None)
            }
        },
        // overwrite nothing, keep as-is
        (DataType::Datetime(_, _), None) => dtype_in.clone(),
        // overwrite time unit, keep timezone
        (DataType::Datetime(_, tz), Some(tu)) => DataType::Datetime(tu, tz.clone()),
        (dt, _) => polars_bail!(InvalidOperation: "expected a temporal datatype, got {}", dt),
    };

    // overwrite time zone, if specified
    match (&dtype, &time_zone) {
        #[cfg(feature = "timezones")]
        (DataType::Datetime(tu, _), Some(tz)) => {
            dtype = DataType::Datetime(*tu, Some(tz.clone()));
        },
        _ => {},
    };

    let (start, end) = if dtype_in == &DataType::Date {
        (
            start.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?,
            end.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?,
        )
    } else {
        (start.clone(), end.clone())
    };

    // If `start` and `end` are naive, but a time zone was specified,
    // then first localize them
    let (start, end) = match (start.dtype(), time_zone) {
        #[cfg(feature = "timezones")]
        (DataType::Datetime(_, None), Some(tz)) => (
            polars_ops::prelude::replace_time_zone(
                start.datetime().unwrap(),
                Some(&tz),
                &StringChunked::from_iter(std::iter::once("raise")),
                NonExistent::Raise,
            )?
            .cast(&dtype)?
            .into_column(),
            polars_ops::prelude::replace_time_zone(
                end.datetime().unwrap(),
                Some(&tz),
                &StringChunked::from_iter(std::iter::once("raise")),
                NonExistent::Raise,
            )?
            .cast(&dtype)?
            .into_column(),
        ),
        _ => (start.cast(&dtype)?, end.cast(&dtype)?),
    };

    let name = start.name();
    let start = temporal_series_to_i64_scalar(&start)
        .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?;
    let end = temporal_series_to_i64_scalar(&end)
        .ok_or_else(|| polars_err!(ComputeError: "end is an out-of-range time."))?;

    let result = match dtype {
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
    Ok(result.cast(&dtype).unwrap().into_column())
}

pub(super) fn datetime_ranges(
    s: &[Column],
    interval: Duration,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> PolarsResult<Column> {
    let mut start = s[0].clone();
    let mut end = s[1].clone();

    // Note: `start` and `end` have already been cast to their supertype,
    // so only `start`'s dtype needs to be matched against.
    #[allow(unused_mut)] // `dtype` is mutated within a "feature = timezones" block
    let mut dtype = match (start.dtype(), time_unit) {
        (DataType::Date, time_unit) => {
            if let Some(tu) = time_unit {
                DataType::Datetime(tu, None)
            } else if interval.nanoseconds() % 1_000 != 0 {
                DataType::Datetime(TimeUnit::Nanoseconds, None)
            } else {
                DataType::Datetime(TimeUnit::Microseconds, None)
            }
        },
        // overwrite nothing, keep as-is
        (DataType::Datetime(_, _), None) => start.dtype().clone(),
        // overwrite time unit, keep timezone
        (DataType::Datetime(_, tz), Some(tu)) => DataType::Datetime(tu, tz.clone()),
        _ => unreachable!(),
    };

    // overwrite time zone, if specified
    match (&dtype, &time_zone) {
        #[cfg(feature = "timezones")]
        (DataType::Datetime(tu, _), Some(tz)) => {
            dtype = DataType::Datetime(*tu, Some(tz.clone()));
        },
        _ => {},
    };

    if start.dtype() == &DataType::Date {
        start = start.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?;
        end = end.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?;
    }

    // If `start` and `end` are naive, but a time zone was specified,
    // then first localize them
    let (start, end) = match (start.dtype(), time_zone) {
        #[cfg(feature = "timezones")]
        (DataType::Datetime(_, None), Some(tz)) => (
            polars_ops::prelude::replace_time_zone(
                start.datetime().unwrap(),
                Some(&tz),
                &StringChunked::from_iter(std::iter::once("raise")),
                NonExistent::Raise,
            )?
            .cast(&dtype)?
            .into_column()
            .to_physical_repr()
            .cast(&DataType::Int64)?,
            polars_ops::prelude::replace_time_zone(
                end.datetime().unwrap(),
                Some(&tz),
                &StringChunked::from_iter(std::iter::once("raise")),
                NonExistent::Raise,
            )?
            .cast(&dtype)?
            .into_column()
            .to_physical_repr()
            .cast(&DataType::Int64)?,
        ),
        _ => (
            start
                .cast(&dtype)?
                .to_physical_repr()
                .cast(&DataType::Int64)?,
            end.cast(&dtype)?
                .to_physical_repr()
                .cast(&DataType::Int64)?,
        ),
    };

    let start = start.i64().unwrap();
    let end = end.i64().unwrap();

    let out = match dtype {
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

    let to_type = DataType::List(Box::new(dtype));
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

        let out_dtype = DataType::Datetime(tu, tz.clone());
        println!("out_dtype: {:?}", out_dtype);
        Ok(DataType::Datetime(tu, tz))
    }
}
