use super::*;

pub(super) fn shift(s: &Series, periods: i64) -> PolarsResult<Series> {
    Ok(s.shift(periods))
}

pub(super) fn reverse(s: &Series) -> PolarsResult<Series> {
    Ok(s.reverse())
}

#[cfg(feature = "approx_unique")]
pub(super) fn approx_unique(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::approx_unique(s)
}

#[cfg(feature = "diff")]
pub(super) fn diff(s: &Series, n: i64, null_behavior: NullBehavior) -> PolarsResult<Series> {
    s.diff(n, null_behavior)
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
pub(super) fn replace_time_zone(
    s: &Series,
    time_zone: Option<&str>,
    use_earliest: Option<bool>,
) -> PolarsResult<Series> {
    let ca = s.datetime().unwrap();
    Ok(polars_ops::prelude::replace_time_zone(ca, time_zone, use_earliest)?.into_series())
}
