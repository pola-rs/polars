use super::*;

pub(super) fn round(s: &Series, decimals: u32) -> PolarsResult<Series> {
    s.round(decimals)
}

pub(super) fn round_sig_figs(s: &Series, digits: i32) -> PolarsResult<Series> {
    s.round_sig_figs(digits)
}

pub(super) fn floor(s: &Series) -> PolarsResult<Series> {
    s.floor()
}

pub(super) fn ceil(s: &Series) -> PolarsResult<Series> {
    s.ceil()
}
