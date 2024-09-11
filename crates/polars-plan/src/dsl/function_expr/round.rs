use super::*;

pub(super) fn round(s: &Column, decimals: u32) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series().round(decimals).map(Column::from)
}

pub(super) fn round_sig_figs(s: &Column, digits: i32) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .round_sig_figs(digits)
        .map(Column::from)
}

pub(super) fn floor(s: &Column) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series().floor().map(Column::from)
}

pub(super) fn ceil(s: &Column) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series().ceil().map(Column::from)
}
