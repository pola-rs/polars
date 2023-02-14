use std::ops::Not;

use super::*;

pub(super) fn shift(s: &Series, periods: i64) -> PolarsResult<Series> {
    Ok(s.shift(periods))
}

pub(super) fn reverse(s: &Series) -> PolarsResult<Series> {
    Ok(s.reverse())
}

pub(super) fn is_null(s: &Series) -> PolarsResult<Series> {
    Ok(s.is_null().into_series())
}

pub(super) fn is_not_null(s: &Series) -> PolarsResult<Series> {
    Ok(s.is_not_null().into_series())
}

pub(super) fn is_not(s: &Series) -> PolarsResult<Series> {
    Ok(s.bool()?.not().into_series())
}

pub(super) fn is_unique(s: &Series) -> PolarsResult<Series> {
    s.is_unique().map(|ca| ca.into_series())
}

pub(super) fn is_duplicated(s: &Series) -> PolarsResult<Series> {
    s.is_duplicated().map(|ca| ca.into_series())
}

#[cfg(feature = "diff")]
pub(super) fn diff(s: &Series, n: usize, null_behavior: NullBehavior) -> PolarsResult<Series> {
    Ok(s.diff(n, null_behavior))
}

#[cfg(feature = "interpolate")]
pub(super) fn interpolate(s: &Series, method: InterpolationMethod) -> PolarsResult<Series> {
    Ok(polars_ops::prelude::interpolate(s, method))
}
#[cfg(feature = "dot_product")]
pub(super) fn dot_impl(s: &[Series]) -> PolarsResult<Series> {
    Ok((&s[0] * &s[1]).sum_as_series())
}

#[cfg(feature = "log")]
pub(super) fn entropy(s: &Series, base: f64, normalize: bool) -> PolarsResult<Series> {
    let out = s.entropy(base, normalize);
    if matches!(s.dtype(), DataType::Float32) {
        let out = out.map(|v| v as f32);
        Ok(Series::new(s.name(), [out]))
    } else {
        Ok(Series::new(s.name(), [out]))
    }
}
