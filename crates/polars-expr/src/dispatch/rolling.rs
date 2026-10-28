use polars_arrow::temporal_conversions::MICROSECONDS_IN_DAY as US_IN_DAY;
use polars_core::error::PolarsResult;
#[cfg(feature = "cov")]
use polars_core::error::{polars_bail, polars_ensure};
use polars_core::prelude::{Column, DataType, IntoColumn, RollingOptionsFixedWindow, TimeUnit};
use polars_core::series::Series;
#[cfg(feature = "cov")]
use polars_plan::dsl::RollingCovOptions;
use polars_plan::prelude::PlanCallback;
use polars_time::prelude::SeriesOpsTime;

fn roll_with_temporal_conversion<F: FnOnce(&Series) -> PolarsResult<Series>>(
    s: &Column,
    op: F,
) -> PolarsResult<Column> {
    let dt = s.dtype();
    let s = if dt.is_temporal() {
        &s.to_physical_repr()
    } else {
        s
    };

    // @scalar-opt
    let out = op(s.as_materialized_series())?;

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

pub(super) fn rolling_min(s: &Column, options: RollingOptionsFixedWindow) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .rolling_min(options)
        .map(Column::from)
}

pub(super) fn rolling_max(s: &Column, options: RollingOptionsFixedWindow) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .rolling_max(options)
        .map(Column::from)
}

pub(super) fn rolling_mean(s: &Column, options: RollingOptionsFixedWindow) -> PolarsResult<Column> {
    roll_with_temporal_conversion(s, |s| s.rolling_mean(options))
}

pub(super) fn rolling_sum(s: &Column, options: RollingOptionsFixedWindow) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .rolling_sum(options)
        .map(Column::from)
}

pub(super) fn rolling_quantile(
    s: &Column,
    options: RollingOptionsFixedWindow,
) -> PolarsResult<Column> {
    roll_with_temporal_conversion(s, |s| s.rolling_quantile(options))
}

pub(super) fn rolling_var(s: &Column, options: RollingOptionsFixedWindow) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .rolling_var(options)
        .map(Column::from)
}

pub(super) fn rolling_std(s: &Column, options: RollingOptionsFixedWindow) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .rolling_std(options)
        .map(Column::from)
}

pub(super) fn rolling_rank(s: &Column, options: RollingOptionsFixedWindow) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .rolling_rank(options)
        .map(Column::from)
}

#[cfg(feature = "moment")]
pub(super) fn rolling_skew(s: &Column, options: RollingOptionsFixedWindow) -> PolarsResult<Column> {
    // @scalar-opt
    let s = s.as_materialized_series();
    polars_ops::series::rolling_skew(s, options).map(Column::from)
}

#[cfg(feature = "moment")]
pub(super) fn rolling_kurtosis(
    s: &Column,
    options: RollingOptionsFixedWindow,
) -> PolarsResult<Column> {
    // @scalar-opt
    let s = s.as_materialized_series();
    polars_ops::series::rolling_kurtosis(s, options).map(Column::from)
}

#[cfg(feature = "cov")]
pub(super) fn rolling_corr_cov(
    s: &[Column],
    rolling_options: RollingOptionsFixedWindow,
    cov_options: RollingCovOptions,
    is_corr: bool,
) -> PolarsResult<Column> {
    use polars_arrow::compute::utils::combine_validities_and;
    use polars_compute::rolling::{rolling_corr, rolling_cov};
    #[cfg(feature = "dtype-f16")]
    use polars_core::datatypes::Float16Type;
    use polars_core::datatypes::{Float32Type, Float64Type};
    use polars_core::with_match_physical_float_polars_type;

    if is_corr && cov_options.ddof.is_some() {
        polars_bail!(InvalidOperation: "options.ddof must be None for rolling_corr");
    }
    let RollingOptionsFixedWindow {
        window_size,
        min_periods,
        ..
    } = rolling_options;
    polars_ensure!(
        min_periods <= window_size,
        InvalidOperation: "`min_periods` ({}) should be <= `window_size` ({})",
        min_periods, window_size
    );

    let st = match polars_core::utils::try_get_supertype(s[0].dtype(), s[1].dtype())? {
        dt if dt.is_float() => dt,
        _ => DataType::Float64,
    };
    let len = if s[0].len() == 1 {
        s[1].len()
    } else {
        s[0].len()
    };
    let prepare = |c: &Column| -> PolarsResult<Series> {
        let c = match c.len() {
            n if n == len => c.clone(),
            1 => c.new_from_index(0, len),
            n => {
                let name = if is_corr {
                    "rolling_corr"
                } else {
                    "rolling_cov"
                };
                polars_bail!(length_mismatch = name, n, len)
            },
        };
        Ok(c.as_materialized_series().cast(&st)?.rechunk())
    };
    let x = prepare(&s[0])?;
    let y = prepare(&s[1])?;

    with_match_physical_float_polars_type!(&st, |$T| {
        let x_arr = x.unpack::<$T>()?.downcast_as_array();
        let y_arr = y.unpack::<$T>()?.downcast_as_array();
        let validity = combine_validities_and(x_arr.validity(), y_arr.validity());
        let (x_vals, y_vals) = (x_arr.values().as_slice(), y_arr.values().as_slice());
        let out = if is_corr {
            rolling_corr(x_vals, y_vals, validity.as_ref(), window_size, min_periods)
        } else {
            let ddof = cov_options.ddof.unwrap_or(1);
            rolling_cov(x_vals, y_vals, validity.as_ref(), window_size, min_periods, ddof)
        };
        Series::try_from((x.name().clone(), out)).map(Column::from)
    })
}

pub fn rolling_map(
    c: &Column,
    rolling_options: RollingOptionsFixedWindow,
    f: PlanCallback<Series, Series>,
) -> PolarsResult<Column> {
    c.as_materialized_series()
        .rolling_map(
            &(|s: &Series| f.call(s.clone())?.strict_cast(s.dtype())) as &_,
            rolling_options,
        )
        .map(Column::from)
}
