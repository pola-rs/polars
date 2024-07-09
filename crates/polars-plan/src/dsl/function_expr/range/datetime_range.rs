#[cfg(feature = "timezones")]
use polars_core::chunked_array::temporal::parse_time_zone;
use polars_core::prelude::*;
use polars_time::{datetime_range_impl, ClosedWindow, Duration};

use super::utils::{
    ensure_range_bounds_contain_exactly_one_value, temporal_ranges_impl_broadcast,
    temporal_series_to_i64_scalar,
};
use crate::dsl::function_expr::FieldsMapper;

const CAPACITY_FACTOR: usize = 5;

pub(super) fn datetime_range(
    s: &[Series],
    interval: Duration,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> PolarsResult<Series> {
    let mut start = s[0].clone();
    let mut end = s[1].clone();

    ensure_range_bounds_contain_exactly_one_value(&start, &end)?;

    // Note: `start` and `end` have already been cast to their supertype,
    // so only `start`'s dtype needs to be matched against.
    #[allow(unused_mut)] // `dtype` is mutated within a "feature = timezones" block.
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
            .into_series(),
            polars_ops::prelude::replace_time_zone(
                end.datetime().unwrap(),
                Some(&tz),
                &StringChunked::from_iter(std::iter::once("raise")),
                NonExistent::Raise,
            )?
            .cast(&dtype)?
            .into_series(),
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
            datetime_range_impl(name, start, end, interval, closed, tu, tz.as_ref())?
        },
        _ => unimplemented!(),
    };
    Ok(result.cast(&dtype).unwrap().into_series())
}

pub(super) fn datetime_ranges(
    s: &[Series],
    interval: Duration,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> PolarsResult<Series> {
    let mut start = s[0].clone();
    let mut end = s[1].clone();

    // Note: `start` and `end` have already been cast to their supertype,
    // so only `start`'s dtype needs to be matched against.
    #[allow(unused_mut)] // `dtype` is mutated within a "feature = timezones" block.
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
            .into_series()
            .to_physical_repr()
            .cast(&DataType::Int64)?,
            polars_ops::prelude::replace_time_zone(
                end.datetime().unwrap(),
                Some(&tz),
                &StringChunked::from_iter(std::iter::once("raise")),
                NonExistent::Raise,
            )?
            .cast(&dtype)?
            .into_series()
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
                start.name(),
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
                let rng = datetime_range_impl("", start, end, interval, closed, tu, tz.as_ref())?;
                builder.append_slice(rng.cont_slice().unwrap());
                Ok(())
            };

            temporal_ranges_impl_broadcast(start, end, range_impl, &mut builder)?
        },
        _ => unimplemented!(),
    };

    let to_type = DataType::List(Box::new(dtype));
    out.cast(&to_type)
}

impl<'a> FieldsMapper<'a> {
    pub(super) fn map_to_datetime_range_dtype(
        &self,
        time_unit: Option<&TimeUnit>,
        time_zone: Option<&str>,
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
        let tz = match time_zone {
            Some(tz) => Some(tz.to_string()),
            None => data_tz,
        };

        Ok(DataType::Datetime(tu, tz))
    }
}
