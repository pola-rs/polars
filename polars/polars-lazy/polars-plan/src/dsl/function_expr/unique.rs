use super::*;

pub(super) fn unique(s: &Series, stable: bool) -> PolarsResult<Series> {
    if stable {
        s.unique_stable()
    } else {
        s.unique()
    }
}

#[cfg(feature = "is_unique")]
pub(super) fn is_unique(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::is_unique(s).map(|ca| ca.into_series())
}

#[cfg(feature = "is_unique")]
pub(super) fn is_duplicated(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::is_duplicated(s).map(|ca| ca.into_series())
}
