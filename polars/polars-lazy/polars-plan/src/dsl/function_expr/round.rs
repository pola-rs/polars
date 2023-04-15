use super::*;

pub(super) fn round(s: &Series, decimals: u32) -> PolarsResult<Series> {
    s.round(decimals)
}

pub(super) fn floor(s: &Series) -> PolarsResult<Series> {
    s.floor()
}

pub(super) fn ceil(s: &Series) -> PolarsResult<Series> {
    s.ceil()
}
