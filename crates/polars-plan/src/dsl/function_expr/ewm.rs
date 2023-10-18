use super::*;

pub(super) fn ewm_mean(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    polars_ops::prelude::ewm_mean(s, options)
}

pub(super) fn ewm_std(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    polars_ops::prelude::ewm_std(s, options)
}

pub(super) fn ewm_var(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    polars_ops::prelude::ewm_var(s, options)
}
