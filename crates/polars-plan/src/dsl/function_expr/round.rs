use super::*;

pub(super) fn round(s: &Series, decimals: u32) -> PolarsResult<Series> {
    s.round(decimals)
}

pub(super) fn round_sf(s: &Series, significant_figures: u32) -> PolarsResult<Series> {
    s.round_sf(significant_figures)
}

pub(super) fn floor(s: &Series) -> PolarsResult<Series> {
    s.floor()
}

pub(super) fn ceil(s: &Series) -> PolarsResult<Series> {
    s.ceil()
}
