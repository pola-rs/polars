use super::*;

pub(super) fn reverse(s: &Column) -> PolarsResult<Column> {
    Ok(s.reverse())
}

#[cfg(feature = "approx_unique")]
pub(super) fn approx_n_unique(s: &Column) -> PolarsResult<Column> {
    s.approx_n_unique()
        .map(|v| Column::new_scalar(s.name().clone(), Scalar::new(IDX_DTYPE, v.into()), 1))
}

#[cfg(feature = "diff")]
pub(super) fn diff(s: &Column, n: i64, null_behavior: NullBehavior) -> PolarsResult<Column> {
    polars_ops::prelude::diff(s.as_materialized_series(), n, null_behavior).map(Column::from)
}

#[cfg(feature = "pct_change")]
pub(super) fn pct_change(s: &[Column]) -> PolarsResult<Column> {
    polars_ops::prelude::pct_change(s[0].as_materialized_series(), s[1].as_materialized_series())
        .map(Column::from)
}

#[cfg(feature = "interpolate")]
pub(super) fn interpolate(s: &Column, method: InterpolationMethod) -> PolarsResult<Column> {
    Ok(polars_ops::prelude::interpolate(s.as_materialized_series(), method).into())
}

#[cfg(feature = "interpolate_by")]
pub(super) fn interpolate_by(s: &[Column]) -> PolarsResult<Column> {
    let by = &s[1];
    let by_is_sorted = by.as_materialized_series().is_sorted(Default::default())?;
    polars_ops::prelude::interpolate_by(&s[0], by, by_is_sorted)
}

pub(super) fn to_physical(s: &Column) -> PolarsResult<Column> {
    Ok(s.to_physical_repr())
}

pub(super) fn set_sorted_flag(s: &Column, sorted: IsSorted) -> PolarsResult<Column> {
    let mut s = s.clone();
    s.set_sorted_flag(sorted);
    Ok(s)
}

#[cfg(feature = "timezones")]
pub(super) fn replace_time_zone(
    s: &[Column],
    time_zone: Option<&str>,
    non_existent: NonExistent,
) -> PolarsResult<Column> {
    let s1 = &s[0];
    let ca = s1.datetime().unwrap();
    let s2 = &s[1].str()?;
    Ok(polars_ops::prelude::replace_time_zone(ca, time_zone, s2, non_existent)?.into_column())
}

#[cfg(feature = "dtype-struct")]
pub(super) fn value_counts(
    s: &Column,
    sort: bool,
    parallel: bool,
    name: PlSmallStr,
    normalize: bool,
) -> PolarsResult<Column> {
    s.as_materialized_series()
        .value_counts(sort, parallel, name, normalize)
        .map(|df| df.into_struct(s.name().clone()).into_column())
}

#[cfg(feature = "unique_counts")]
pub(super) fn unique_counts(s: &Column) -> PolarsResult<Column> {
    polars_ops::prelude::unique_counts(s.as_materialized_series()).map(Column::from)
}

#[cfg(feature = "dtype-array")]
pub(super) fn reshape(c: &Column, dimensions: &[ReshapeDimension]) -> PolarsResult<Column> {
    c.reshape_array(dimensions)
}

#[cfg(feature = "repeat_by")]
pub(super) fn repeat_by(s: &[Column]) -> PolarsResult<Column> {
    let by = &s[1];
    let s = &s[0];
    let by = by.cast(&IDX_DTYPE)?;
    polars_ops::chunked_array::repeat_by(s.as_materialized_series(), by.idx()?)
        .map(|ok| ok.into_column())
}

pub(super) fn backward_fill(s: &Column, limit: FillNullLimit) -> PolarsResult<Column> {
    s.fill_null(FillNullStrategy::Backward(limit))
}

pub(super) fn forward_fill(s: &Column, limit: FillNullLimit) -> PolarsResult<Column> {
    s.fill_null(FillNullStrategy::Forward(limit))
}

pub(super) fn max_horizontal(s: &mut [Column]) -> PolarsResult<Option<Column>> {
    polars_ops::prelude::max_horizontal(s)
}

pub(super) fn min_horizontal(s: &mut [Column]) -> PolarsResult<Option<Column>> {
    polars_ops::prelude::min_horizontal(s)
}

pub(super) fn sum_horizontal(s: &mut [Column]) -> PolarsResult<Option<Column>> {
    polars_ops::prelude::sum_horizontal(s)
}

pub(super) fn mean_horizontal(s: &mut [Column]) -> PolarsResult<Option<Column>> {
    polars_ops::prelude::mean_horizontal(s)
}

pub(super) fn drop_nulls(s: &Column) -> PolarsResult<Column> {
    Ok(s.drop_nulls())
}

#[cfg(feature = "mode")]
pub(super) fn mode(s: &Column) -> PolarsResult<Column> {
    mode::mode(s.as_materialized_series()).map(Column::from)
}

#[cfg(feature = "moment")]
pub(super) fn skew(s: &Column, bias: bool) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .skew(bias)
        .map(|opt_v| Column::new(s.name().clone(), &[opt_v]))
}

#[cfg(feature = "moment")]
pub(super) fn kurtosis(s: &Column, fisher: bool, bias: bool) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .kurtosis(fisher, bias)
        .map(|opt_v| Column::new(s.name().clone(), &[opt_v]))
}

pub(super) fn arg_unique(s: &Column) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .arg_unique()
        .map(|ok| ok.into_column())
}

#[cfg(feature = "rank")]
pub(super) fn rank(s: &Column, options: RankOptions, seed: Option<u64>) -> PolarsResult<Column> {
    Ok(s.as_materialized_series().rank(options, seed).into_column())
}

#[cfg(feature = "hist")]
pub(super) fn hist(
    s: &[Column],
    bin_count: Option<usize>,
    include_category: bool,
    include_breakpoint: bool,
) -> PolarsResult<Column> {
    let bins = if s.len() == 2 { Some(&s[1]) } else { None };
    let s = s[0].as_materialized_series();
    hist_series(
        s,
        bin_count,
        bins.map(|b| b.as_materialized_series().clone()),
        include_category,
        include_breakpoint,
    )
    .map(Column::from)
}

#[cfg(feature = "replace")]
pub(super) fn replace(s: &[Column]) -> PolarsResult<Column> {
    polars_ops::series::replace(
        s[0].as_materialized_series(),
        s[1].as_materialized_series(),
        s[2].as_materialized_series(),
    )
    .map(Column::from)
}

#[cfg(feature = "replace")]
pub(super) fn replace_strict(s: &[Column], return_dtype: Option<DataType>) -> PolarsResult<Column> {
    match s.get(3) {
        Some(default) => polars_ops::series::replace_or_default(
            s[0].as_materialized_series(),
            s[1].as_materialized_series(),
            s[2].as_materialized_series(),
            default.as_materialized_series(),
            return_dtype,
        ),
        None => polars_ops::series::replace_strict(
            s[0].as_materialized_series(),
            s[1].as_materialized_series(),
            s[2].as_materialized_series(),
            return_dtype,
        ),
    }
    .map(Column::from)
}

pub(super) fn fill_null_with_strategy(
    s: &Column,
    strategy: FillNullStrategy,
) -> PolarsResult<Column> {
    s.fill_null(strategy)
}

pub(super) fn gather_every(s: &Column, n: usize, offset: usize) -> PolarsResult<Column> {
    polars_ensure!(n > 0, InvalidOperation: "gather_every(n): n should be positive");
    Ok(s.gather_every(n, offset))
}

#[cfg(feature = "reinterpret")]
pub(super) fn reinterpret(s: &Column, signed: bool) -> PolarsResult<Column> {
    polars_ops::series::reinterpret(s.as_materialized_series(), signed).map(Column::from)
}

pub(super) fn negate(s: &Column) -> PolarsResult<Column> {
    polars_ops::series::negate(s.as_materialized_series()).map(Column::from)
}

pub(super) fn extend_constant(s: &[Column]) -> PolarsResult<Column> {
    let value = &s[1];
    let n = &s[2];
    polars_ensure!(value.len() == 1 && n.len() == 1, ComputeError: "value and n should have unit length.");
    let n = n.strict_cast(&DataType::UInt64)?;
    let v = value.get(0)?;
    let s = &s[0];
    match n.u64()?.get(0) {
        Some(n) => s.extend_constant(v, n as usize),
        None => {
            polars_bail!(ComputeError: "n can not be None for extend_constant.")
        },
    }
}
