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

/// Datetime range given start, end, and interval.
fn dt_range_start_end_interval(
    start: &Column,
    end: &Column,
    interval: Duration,
    closed: ClosedWindow,
    // time_unit: Option<TimeUnit>,
    // time_zone: Option<TimeZone>,
) -> PolarsResult<Column> {
    ensure_items_contain_exactly_one_value(&[&start, &end], &["start", "end"])?;
    let dtype = start.dtype();

    let name = start.name();
    let start = temporal_series_to_i64_scalar(&start)
        .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?;
    let end = temporal_series_to_i64_scalar(&end)
        .ok_or_else(|| polars_err!(ComputeError: "end is an out-of-range time."))?;

    if let DataType::Datetime(tu, time_zone) = dtype {
        let tz = match time_zone {
            #[cfg(feature = "timezones")]
            Some(tz) => Some(parse_time_zone(&tz)?),
            _ => None,
        };
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
        polars_bail!(ComputeError: "nope");
    }
}

fn dt_ranges_start_end_interval(
    start: &Column,
    end: &Column,
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<Column> {
    ensure_items_contain_exactly_one_value(&[&start, &end], &["start", "end"])?;
    let dtype = start.dtype();

    let name = start.name();
    let start = temporal_series_to_i64_scalar(&start)
        .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?;
    let end = temporal_series_to_i64_scalar(&end)
        .ok_or_else(|| polars_err!(ComputeError: "end is an out-of-range time."))?;

    if let DataType::Datetime(tu, time_zone) = dtype {
        let tz = match time_zone {
            #[cfg(feature = "timezones")]
            Some(tz) => Some(parse_time_zone(&tz)?),
            _ => None,
        };
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
        polars_bail!(ComputeError: "nope");
    }
}

fn dt_range_start_end_samples(
    start: &Column,
    end: &Column,
    num_samples: &Column,
    closed: ClosedWindow,
) -> PolarsResult<Column> {
    ensure_items_contain_exactly_one_value(&[&start, &end], &["start", "end"])?;
    ensure_items_contain_exactly_one_value(&[&start, &end], &["start", "end"])?;
    let dtype = start.dtype();

    let name = start.name();
    let start = temporal_series_to_i64_scalar(&start)
        .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?;
    let end = temporal_series_to_i64_scalar(&end)
        .ok_or_else(|| polars_err!(ComputeError: "end is an out-of-range time."))?;
    let num_samples = num_samples.get(0).unwrap().extract::<u64>().unwrap();

    if let DataType::Datetime(tu, time_zone) = dtype {
        let tz = match time_zone {
            #[cfg(feature = "timezones")]
            Some(tz) => Some(parse_time_zone(&tz)?),
            _ => None,
        };
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
        polars_bail!(ComputeError: "nope");
    }
}

fn dt_range_start_interval_samples(
    start: &Column,
    interval: Duration,
    num_samples: &Column,
    closed: ClosedWindow,
) -> PolarsResult<Column> {
    ensure_items_contain_exactly_one_value(&[&start, &num_samples], &["start", "num_samples"])?;
    let dtype = start.dtype();

    let name = start.name();
    let start = temporal_series_to_i64_scalar(&start)
        .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?;
    let num_samples = num_samples.get(0).unwrap().extract::<u64>().unwrap();

    if let DataType::Datetime(tu, time_zone) = dtype {
        let tz = match time_zone {
            #[cfg(feature = "timezones")]
            Some(tz) => Some(parse_time_zone(&tz)?),
            _ => None,
        };
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
        polars_bail!(ComputeError: "nope");
    }
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
        ),
        DateRangeArgs::StartEndSamples => {
            dt_range_start_end_samples(&s[0].cast(&dt_type)?, &s[1].cast(&dt_type)?, &s[2], closed)
        },
        DateRangeArgs::StartIntervalSamples => {
            dt_range_start_interval_samples(&s[0].cast(&dt_type)?, interval.unwrap(), &s[1], closed)
        },
        // We negate the interval, start at the end, and then reverse.
        DateRangeArgs::EndIntervalSamples => dt_range_start_interval_samples(
            &s[0].cast(&dt_type)?,
            -interval.unwrap(),
            &s[1],
            closed,
        ),
    }
    .map(|c| c.cast(&DataType::Date))?
}

pub(super) fn datetime_range(
    s: &[Column],
    interval: Option<Duration>,
    closed: ClosedWindow,
    // time_unit: Option<TimeUnit>,
    // time_zone: Option<TimeZone>,
    arg_type: DateRangeArgs,
) -> PolarsResult<Column> {
    match arg_type {
        DateRangeArgs::StartEndInterval => dt_range_start_end_interval(
            &s[0],
            &s[1],
            interval.unwrap(),
            closed,
            // time_unit,
            // time_zone.clone(),
        ),
        DateRangeArgs::StartEndSamples => {
            dt_range_start_end_samples(
                &s[0], &s[1], &s[2], closed,
                // time_unit, time_zone.clone(),
            )
        },
        DateRangeArgs::StartIntervalSamples => dt_range_start_interval_samples(
            &s[0],
            interval.unwrap(),
            &s[1],
            closed,
            // time_unit,
            // time_zone.clone(),
        ),
        // We negate the interval, start at the end, and then reverse.
        DateRangeArgs::EndIntervalSamples => dt_range_start_interval_samples(
            &s[0],
            -interval.unwrap(),
            &s[1],
            closed,
            // time_unit,
            // time_zone.clone(),
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
    let mut start = s[0].clone();
    let mut end = s[1].clone();

    // Note: `start` and `end` have already been cast to their supertype,
    // so only `start`'s dtype needs to be matched against.
    #[allow(unused_mut)] // `dtype` is mutated within a "feature = timezones" block.
    let mut dtype = match (start.dtype(), time_unit) {
        (DataType::Date, time_unit) => {
            if let Some(tu) = time_unit {
                DataType::Datetime(tu, None)
            } else if interval.unwrap().nanoseconds() % 1_000 != 0 {
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
                    interval.unwrap(),
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

        Ok(DataType::Datetime(tu, tz))
    }
}
