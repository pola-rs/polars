use super::*;

pub(super) fn reverse(s: &Series) -> PolarsResult<Series> {
    Ok(s.reverse())
}

#[cfg(feature = "approx_unique")]
pub(super) fn approx_n_unique(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::approx_n_unique(s)
}

#[cfg(feature = "diff")]
pub(super) fn diff(s: &Series, n: i64, null_behavior: NullBehavior) -> PolarsResult<Series> {
    polars_ops::prelude::diff(s, n, null_behavior)
}

#[cfg(feature = "pct_change")]
pub(super) fn pct_change(s: &[Series]) -> PolarsResult<Series> {
    polars_ops::prelude::pct_change(&s[0], &s[1])
}

#[cfg(feature = "interpolate")]
pub(super) fn interpolate(s: &Series, method: InterpolationMethod) -> PolarsResult<Series> {
    Ok(polars_ops::prelude::interpolate(s, method))
}

pub(super) fn to_physical(s: &Series) -> PolarsResult<Series> {
    Ok(s.to_physical_repr().into_owned())
}

pub(super) fn set_sorted_flag(s: &Series, sorted: IsSorted) -> PolarsResult<Series> {
    let mut s = s.clone();
    s.set_sorted_flag(sorted);
    Ok(s)
}

#[cfg(feature = "timezones")]
pub(super) fn replace_time_zone(s: &[Series], time_zone: Option<&str>) -> PolarsResult<Series> {
    let s1 = &s[0];
    let ca = s1.datetime().unwrap();
    let s2 = &s[1].utf8().unwrap();
    Ok(polars_ops::prelude::replace_time_zone(ca, time_zone, s2)?.into_series())
}

#[cfg(feature = "timezones")]
pub(super) fn convert_and_replace_time_zone(
    s: &[Series],
    time_zone: Option<&str>,
) -> PolarsResult<Series> {
    let s1 = &s[0];
    let ca = s1.datetime().unwrap();
    let s2 = &s[1].utf8().unwrap();
    Ok(polars_ops::prelude::convert_and_replace_time_zone(ca, time_zone, s2)?.into_series())
}

#[cfg(feature = "dtype-struct")]
pub(super) fn value_counts(s: &Series, sort: bool, parallel: bool) -> PolarsResult<Series> {
    s.value_counts(sort, parallel)
        .map(|df| df.into_struct(s.name()).into_series())
}

#[cfg(feature = "unique_counts")]
pub(super) fn unique_counts(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::unique_counts(s)
}

pub(super) fn reshape(s: &Series, dims: Vec<i64>) -> PolarsResult<Series> {
    s.reshape(&dims)
}

#[cfg(feature = "repeat_by")]
pub(super) fn repeat_by(s: &[Series]) -> PolarsResult<Series> {
    let by = &s[1];
    let s = &s[0];
    let by = by.cast(&IDX_DTYPE)?;
    polars_ops::chunked_array::repeat_by(s, by.idx()?).map(|ok| ok.into_series())
}

pub(super) fn backward_fill(s: &Series, limit: FillNullLimit) -> PolarsResult<Series> {
    s.fill_null(FillNullStrategy::Backward(limit))
}

pub(super) fn forward_fill(s: &Series, limit: FillNullLimit) -> PolarsResult<Series> {
    s.fill_null(FillNullStrategy::Forward(limit))
}

pub(super) fn sum_horizontal(s: &[Series]) -> PolarsResult<Series> {
    polars_ops::prelude::sum_horizontal(s)
}

pub(super) fn max_horizontal(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    polars_ops::prelude::max_horizontal(s)
}

pub(super) fn min_horizontal(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    polars_ops::prelude::min_horizontal(s)
}

pub(super) fn drop_nulls(s: &Series) -> PolarsResult<Series> {
    Ok(s.drop_nulls())
}

#[cfg(feature = "mode")]
pub(super) fn mode(s: &Series) -> PolarsResult<Series> {
    mode::mode(s)
}

#[cfg(feature = "moment")]
pub(super) fn skew(s: &Series, bias: bool) -> PolarsResult<Series> {
    s.skew(bias).map(|opt_v| Series::new(s.name(), &[opt_v]))
}

#[cfg(feature = "moment")]
pub(super) fn kurtosis(s: &Series, fisher: bool, bias: bool) -> PolarsResult<Series> {
    s.kurtosis(fisher, bias)
        .map(|opt_v| Series::new(s.name(), &[opt_v]))
}

pub(super) fn arg_unique(s: &Series) -> PolarsResult<Series> {
    s.arg_unique().map(|ok| ok.into_series())
}

#[cfg(feature = "rank")]
pub(super) fn rank(s: &Series, options: RankOptions, seed: Option<u64>) -> PolarsResult<Series> {
    Ok(s.rank(options, seed))
}
