use polars_core::prelude::*;
use polars_core::series::Series;
use polars_core::utils::arrow::temporal_conversions::MILLISECONDS_IN_DAY;
use polars_time::{datetime_range_impl, ClosedWindow, Duration};

use super::datetime_range::{datetime_range, datetime_ranges};
use super::utils::{
    ensure_range_bounds_contain_exactly_one_value, temporal_ranges_impl_broadcast,
    temporal_series_to_i64_scalar,
};
use crate::dsl::function_expr::FieldsMapper;

const CAPACITY_FACTOR: usize = 5;

pub(super) fn temporal_range(
    s: &[Series],
    interval: Duration,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> PolarsResult<Series> {
    if s[0].dtype() == &DataType::Date && interval.is_full_days() {
        date_range(s, interval, closed)
    } else {
        datetime_range(s, interval, closed, time_unit, time_zone)
    }
}

pub(super) fn temporal_ranges(
    s: &[Series],
    interval: Duration,
    closed: ClosedWindow,
    time_unit: Option<TimeUnit>,
    time_zone: Option<TimeZone>,
) -> PolarsResult<Series> {
    if s[0].dtype() == &DataType::Date && interval.is_full_days() {
        date_ranges(s, interval, closed)
    } else {
        datetime_ranges(s, interval, closed, time_unit, time_zone)
    }
}

fn date_range(s: &[Series], interval: Duration, closed: ClosedWindow) -> PolarsResult<Series> {
    let start = &s[0];
    let end = &s[1];
    let name = start.name();

    ensure_range_bounds_contain_exactly_one_value(start, end)?;

    let dtype = DataType::Date;
    let start = temporal_series_to_i64_scalar(start)
        .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?
        * MILLISECONDS_IN_DAY;
    let end = temporal_series_to_i64_scalar(end)
        .ok_or_else(|| polars_err!(ComputeError: "end is an out-of-range time."))?
        * MILLISECONDS_IN_DAY;

    let result = datetime_range_impl(
        name,
        start,
        end,
        interval,
        closed,
        TimeUnit::Milliseconds,
        None,
    )?
    .cast(&dtype)?;

    Ok(result.into_series())
}

fn date_ranges(s: &[Series], interval: Duration, closed: ClosedWindow) -> PolarsResult<Series> {
    let start = &s[0];
    let end = &s[1];

    let start = start.cast(&DataType::Int64)?;
    let end = end.cast(&DataType::Int64)?;

    let start = start.i64().unwrap() * MILLISECONDS_IN_DAY;
    let end = end.i64().unwrap() * MILLISECONDS_IN_DAY;

    let mut builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(
        start.name(),
        start.len(),
        start.len() * CAPACITY_FACTOR,
        DataType::Int32,
    );

    let range_impl = |start, end, builder: &mut ListPrimitiveChunkedBuilder<Int32Type>| {
        let rng = datetime_range_impl(
            "",
            start,
            end,
            interval,
            closed,
            TimeUnit::Milliseconds,
            None,
        )?;
        let rng = rng.cast(&DataType::Date).unwrap();
        let rng = rng.to_physical_repr();
        let rng = rng.i32().unwrap();
        builder.append_slice(rng.cont_slice().unwrap());
        Ok(())
    };

    let out = temporal_ranges_impl_broadcast(&start, &end, range_impl, &mut builder)?;

    let to_type = DataType::List(Box::new(DataType::Date));
    out.cast(&to_type)
}

impl<'a> FieldsMapper<'a> {
    pub(super) fn map_to_date_range_dtype(
        &self,
        interval: &Duration,
        time_unit: Option<&TimeUnit>,
        time_zone: Option<&str>,
    ) -> PolarsResult<DataType> {
        let data_dtype = self.map_to_supertype()?.dtype;
        match data_dtype {
            DataType::Datetime(tu, tz) => {
                map_datetime_to_date_range_dtype(tu, tz, time_unit, time_zone)
            },
            DataType::Date => {
                let schema_dtype = map_date_to_date_range_dtype(interval, time_unit, time_zone);
                Ok(schema_dtype)
            },
            _ => polars_bail!(ComputeError: "expected Date or Datetime, got {}", data_dtype),
        }
    }
}

fn map_datetime_to_date_range_dtype(
    data_time_unit: TimeUnit,
    data_time_zone: Option<String>,
    given_time_unit: Option<&TimeUnit>,
    given_time_zone: Option<&str>,
) -> PolarsResult<DataType> {
    let schema_time_zone = match (data_time_zone, given_time_zone) {
        (Some(data_tz), Some(given_tz)) => {
            polars_ensure!(
                data_tz == given_tz,
                ComputeError: format!(
                    "`time_zone` does not match the data\
                    \n\nData has time zone '{}', got '{}'.", data_tz, given_tz)
            );
            Some(data_tz)
        },
        (_, Some(given_tz)) => Some(given_tz.to_string()),
        (Some(data_tz), None) => Some(data_tz),
        (_, _) => None,
    };
    let schema_time_unit = given_time_unit.unwrap_or(&data_time_unit);

    let schema_dtype = DataType::Datetime(*schema_time_unit, schema_time_zone);
    Ok(schema_dtype)
}
fn map_date_to_date_range_dtype(
    interval: &Duration,
    time_unit: Option<&TimeUnit>,
    time_zone: Option<&str>,
) -> DataType {
    if interval.is_full_days() {
        DataType::Date
    } else if let Some(tu) = time_unit {
        DataType::Datetime(*tu, time_zone.map(String::from))
    } else if interval.nanoseconds() % 1000 != 0 {
        DataType::Datetime(TimeUnit::Nanoseconds, time_zone.map(String::from))
    } else {
        DataType::Datetime(TimeUnit::Microseconds, time_zone.map(String::from))
    }
}
