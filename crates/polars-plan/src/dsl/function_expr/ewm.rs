use super::*;

pub(super) fn ewm_mean(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    s.ewm_mean(options)
}

pub(super) fn ewm_std(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    s.ewm_std(options)
}

pub(super) fn ewm_var(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    s.ewm_var(options)
}
