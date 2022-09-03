use std::ops::Not;

use super::*;

pub(super) fn shift(s: &Series, periods: i64) -> Result<Series> {
    Ok(s.shift(periods))
}

pub(super) fn reverse(s: &Series) -> Result<Series> {
    Ok(s.reverse())
}

pub(super) fn is_null(s: &Series) -> Result<Series> {
    Ok(s.is_null().into_series())
}

pub(super) fn is_not_null(s: &Series) -> Result<Series> {
    Ok(s.is_not_null().into_series())
}

pub(super) fn is_not(s: &Series) -> Result<Series> {
    Ok(s.bool()?.not().into_series())
}

pub(super) fn is_unique(s: &Series) -> Result<Series> {
    s.is_unique().map(|ca| ca.into_series())
}

pub(super) fn is_duplicated(s: &Series) -> Result<Series> {
    s.is_duplicated().map(|ca| ca.into_series())
}
