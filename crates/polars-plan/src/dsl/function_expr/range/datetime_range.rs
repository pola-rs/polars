use polars_core::prelude::*;
use polars_core::series::Series;
use polars_time::{datetime_range_impl, ClosedWindow, Duration};

use super::utils::{ensure_range_bounds_contain_exactly_one_value, temporal_series_to_i64_scalar};
use crate::dsl::function_expr::FieldsMapper;

const CAPACITY_FACTOR: usize = 5;

pub(super) fn datetime_range(
    s: &[Series],
    interval: Duration,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> PolarsResult<Series> {
    let start = &s[0];
    let end = &s[1];

    ensure_range_bounds_contain_exactly_one_value(start, end)?;

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

    let (start, end) = match dtype {
        #[cfg(feature = "timezones")]
        DataType::Datetime(_, Some(_)) => (
            polars_ops::prelude::replace_time_zone(
                start.cast(&dtype)?.datetime().unwrap(),
                None,
                &Utf8Chunked::from_iter(std::iter::once("raise")),
            )?
            .into_series(),
            polars_ops::prelude::replace_time_zone(
                end.cast(&dtype)?.datetime().unwrap(),
                None,
                &Utf8Chunked::from_iter(std::iter::once("raise")),
            )?
            .into_series(),
        ),
        _ => (start.cast(&dtype)?, end.cast(&dtype)?),
    };

    // overwrite time zone, if specified
    match (&dtype, &time_zone) {
        #[cfg(feature = "timezones")]
        (DataType::Datetime(tu, _), Some(tz)) => {
            dtype = DataType::Datetime(*tu, Some(tz.clone()));
        },
        _ => {},
    };

    let start = temporal_series_to_i64_scalar(&start);
    let end = temporal_series_to_i64_scalar(&end);

    let result = match dtype {
        DataType::Datetime(tu, ref tz) => {
            datetime_range_impl("datetime", start, end, interval, closed, tu, tz.as_ref())?
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
    let start = &s[0];
    let end = &s[1];

    polars_ensure!(
        start.len() == end.len(),
        ComputeError: "`start` and `end` must have the same length",
    );

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

    let (start, end) = match dtype {
        #[cfg(feature = "timezones")]
        DataType::Datetime(_, Some(_)) => (
            polars_ops::prelude::replace_time_zone(
                start.cast(&dtype)?.datetime().unwrap(),
                None,
                &Utf8Chunked::from_iter(std::iter::once("raise")),
            )?
            .into_series()
            .to_physical_repr()
            .cast(&DataType::Int64)?,
            polars_ops::prelude::replace_time_zone(
                end.cast(&dtype)?.datetime().unwrap(),
                None,
                &Utf8Chunked::from_iter(std::iter::once("raise")),
            )?
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

    // overwrite time zone, if specified
    match (&dtype, &time_zone) {
        #[cfg(feature = "timezones")]
        (DataType::Datetime(tu, _), Some(tz)) => {
            dtype = DataType::Datetime(*tu, Some(tz.clone()));
        },
        _ => {},
    };

    let start = start.i64().unwrap();
    let end = end.i64().unwrap();

    let list = match dtype {
        DataType::Datetime(tu, ref tz) => {
            let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
                "datetime_range",
                start.len(),
                start.len() * CAPACITY_FACTOR,
                DataType::Int64,
            );
            for (start, end) in start.into_iter().zip(end) {
                match (start, end) {
                    (Some(start), Some(end)) => {
                        let rng =
                            datetime_range_impl("", start, end, interval, closed, tu, tz.as_ref())?;
                        builder.append_slice(rng.cont_slice().unwrap())
                    },
                    _ => builder.append_null(),
                }
            }
            builder.finish().into_series()
        },
        _ => unimplemented!(),
    };

    let to_type = DataType::List(Box::new(dtype));
    list.cast(&to_type)
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
