use arrow::temporal_conversions::MICROSECONDS_IN_DAY as US_IN_DAY;
use polars_core::error::PolarsResult;
use polars_core::prelude::{Column, DataType, IntoColumn, TimeUnit};
use polars_core::series::Series;
use polars_time::prelude::{RollingOptionsDynamicWindow, SeriesOpsTime};

fn roll_by_with_temporal_conversion<F: FnOnce(&Series, &Series) -> PolarsResult<Series>>(
    s: &[Column],
    op: F,
) -> PolarsResult<Column> {
    let by = &s[1];
    let dt = &s[0].dtype();
    let s = if dt.is_temporal() {
        &s[0].to_physical_repr()
    } else {
        &s[0]
    };

    // @scalar-opt
    let out = op(s.as_materialized_series(), by.as_materialized_series())?;

    Ok(match dt {
        DataType::Date => (out * US_IN_DAY as f64)
            .cast(&DataType::Int64)?
            .into_datetime(TimeUnit::Microseconds, None),
        DataType::Datetime(tu, tz) => out.cast(&DataType::Int64)?.into_datetime(*tu, tz.clone()),
        DataType::Duration(tu) => out.cast(&DataType::Int64)?.into_duration(*tu),
        DataType::Time => out.cast(&DataType::Int64)?.into_time(),
        _ => out,
    }
    .into_column())
}

pub(super) fn rolling_min_by(
    s: &[Column],
    options: RollingOptionsDynamicWindow,
) -> PolarsResult<Column> {
    // @scalar-opt
    s[0].as_materialized_series()
        .rolling_min_by(s[1].as_materialized_series(), options)
        .map(Column::from)
}

pub(super) fn rolling_max_by(
    s: &[Column],
    options: RollingOptionsDynamicWindow,
) -> PolarsResult<Column> {
    // @scalar-opt
    s[0].as_materialized_series()
        .rolling_max_by(s[1].as_materialized_series(), options)
        .map(Column::from)
}

pub(super) fn rolling_mean_by(
    s: &[Column],
    options: RollingOptionsDynamicWindow,
) -> PolarsResult<Column> {
    roll_by_with_temporal_conversion(s, |s, by| s.rolling_mean_by(by, options))
}

pub(super) fn rolling_sum_by(
    s: &[Column],
    options: RollingOptionsDynamicWindow,
) -> PolarsResult<Column> {
    // @scalar-opt
    s[0].as_materialized_series()
        .rolling_sum_by(s[1].as_materialized_series(), options)
        .map(Column::from)
}

pub(super) fn rolling_quantile_by(
    s: &[Column],
    options: RollingOptionsDynamicWindow,
) -> PolarsResult<Column> {
    roll_by_with_temporal_conversion(s, |s, by| s.rolling_quantile_by(by, options))
}

pub(super) fn rolling_var_by(
    s: &[Column],
    options: RollingOptionsDynamicWindow,
) -> PolarsResult<Column> {
    // @scalar-opt
    s[0].as_materialized_series()
        .rolling_var_by(s[1].as_materialized_series(), options)
        .map(Column::from)
}

pub(super) fn rolling_std_by(
    s: &[Column],
    options: RollingOptionsDynamicWindow,
) -> PolarsResult<Column> {
    // @scalar-opt
    s[0].as_materialized_series()
        .rolling_std_by(s[1].as_materialized_series(), options)
        .map(Column::from)
}

pub(super) fn rolling_rank_by(
    s: &[Column],
    options: RollingOptionsDynamicWindow,
) -> PolarsResult<Column> {
    // @scalar-opt
    s[0].as_materialized_series()
        .rolling_rank_by(s[1].as_materialized_series(), options)
        .map(Column::from)
}
