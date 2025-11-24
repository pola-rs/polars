use arrow::temporal_conversions::MICROSECONDS_IN_DAY as US_IN_DAY;
use polars_core::error::PolarsResult;
use polars_core::prelude::{Column, DataType, IntoColumn, TimeUnit};
use polars_time::prelude::{RollingOptionsDynamicWindow, SeriesOpsTime};

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
    // @scalar-opt
    s[0].as_materialized_series()
        .rolling_mean_by(s[1].as_materialized_series(), options)
        .map(Column::from)
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
    let by = &s[1];

    let dt = &s[0].dtype();
    let s = if dt.is_temporal() {
        &s[0].to_physical_repr()
    } else {
        &s[0]
    };

    // @scalar-opt
    let out = s
        .as_materialized_series()
        .rolling_quantile_by(by.as_materialized_series(), options)?;

    Ok(match dt {
        DataType::Date => {
            (out * US_IN_DAY).cast(&DataType::Datetime(TimeUnit::Microseconds, None))?
        },
        DataType::Datetime(tu, tz) => out.cast(&DataType::Datetime(*tu, tz.clone()))?,
        DataType::Duration(tu) => out.cast(&DataType::Duration(*tu))?,
        DataType::Time => out.cast(&DataType::Time)?,
        _ => out,
    }
    .into_column())
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
