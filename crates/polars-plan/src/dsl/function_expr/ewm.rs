use super::*;

#[cfg(feature = "ewma")]
pub(super) fn ewm_mean(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    s.ewm_mean(options)
}

#[cfg(feature = "ewma")]
pub(super) fn ewm_std(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    s.ewm_std(options)
}

#[cfg(feature = "ewma")]
pub(super) fn ewm_var(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    s.ewm_var(options)
}
