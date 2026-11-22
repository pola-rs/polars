use polars_arrow::temporal_conversions::MICROSECONDS_IN_DAY as US_IN_DAY;
use polars_core::error::PolarsResult;
use polars_core::prelude::{ChunkApply, Column, DataType, IntoColumn, IntoSeries, TimeUnit};
use polars_core::series::Series;
use polars_defs::time::rolling::RollingOptionsDynamicWindow;
use polars_time::prelude::SeriesOpsTime;

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

    let out = if matches!(dt, DataType::Date) {
        out * US_IN_DAY as f64
    } else {
        out
    };
    let out = if dt.is_temporal() {
        let ca = out.f64()?;
        if matches!(dt, DataType::Duration(_)) {
            ca.apply_values(f64::trunc).into_series()
        } else {
            ca.apply_values(f64::floor).into_series()
        }
    } else {
        out
    };

    Ok(match dt {
        DataType::Date => out
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
    let dt = s[0].dtype();
    if dt.is_temporal() {
        let physical = if matches!(dt, DataType::Date) {
            s[0].to_physical_repr().cast(&DataType::Int64)?
        } else {
            s[0].to_physical_repr()
        };
        let out = physical.as_materialized_series().rolling_mean_temporal_by(
            s[1].as_materialized_series(),
            options,
            !matches!(dt, DataType::Duration(_)),
            matches!(dt, DataType::Date),
        )?;
        return Ok(match dt {
            DataType::Date => out.into_datetime(TimeUnit::Microseconds, None),
            DataType::Datetime(tu, tz) => out.into_datetime(*tu, tz.clone()),
            DataType::Duration(tu) => out.into_duration(*tu),
            DataType::Time => out.into_time(),
            _ => unreachable!(),
        }
        .into_column());
    }
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
