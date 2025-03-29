use polars_ops::series::round::RoundMode;

use super::*;

pub(super) fn round(c: &Column, decimals: u32, mode: RoundMode) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(|s| s.round(decimals, mode))
}

pub(super) fn round_sig_figs(c: &Column, digits: i32) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(|s| s.round_sig_figs(digits))
}

pub(super) fn floor(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(Series::floor)
}

pub(super) fn ceil(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(Series::ceil)
}
