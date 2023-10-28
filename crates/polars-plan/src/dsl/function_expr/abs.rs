use super::*;

pub(super) fn abs(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::abs(s)
}
