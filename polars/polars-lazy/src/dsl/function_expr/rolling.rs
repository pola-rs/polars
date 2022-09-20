use super::*;

pub(super) fn rolling_skew(s: &Series, window_size: usize, bias: bool) -> PolarsResult<Series> {
    s.rolling_skew(window_size, bias)
}

pub(super) fn rolling_kurtosis(
    s: &Series,
    window_size: usize,
    fisher: bool,
    bias: bool,
) -> PolarsResult<Series> {
    s.rolling_kurtosis(window_size, fisher, bias)
}
