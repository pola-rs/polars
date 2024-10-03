use polars_core::frame::column::ScalarColumn;

use super::*;

pub(super) fn round(c: &Column, decimals: u32) -> PolarsResult<Column> {
    match c {
        Column::Series(s) => s.round(decimals).map(Column::from),
        Column::Scalar(s) if s.is_empty() => {
            s.as_materialized_series().round(decimals).map(Column::from)
        },
        Column::Scalar(s) => Ok(ScalarColumn::from_single_value_series(
            s.as_single_value_series().round(decimals)?,
            s.len(),
        )
        .into()),
    }
}

pub(super) fn round_sig_figs(c: &Column, digits: i32) -> PolarsResult<Column> {
    match c {
        Column::Series(s) => s.round_sig_figs(digits).map(Column::from),
        Column::Scalar(s) if s.is_empty() => s
            .as_materialized_series()
            .round_sig_figs(digits)
            .map(Column::from),
        Column::Scalar(s) => Ok(ScalarColumn::from_single_value_series(
            s.as_single_value_series().round_sig_figs(digits)?,
            s.len(),
        )
        .into()),
    }
}

pub(super) fn floor(c: &Column) -> PolarsResult<Column> {
    match c {
        Column::Series(s) => s.floor().map(Column::from),
        Column::Scalar(s) if s.is_empty() => s.as_materialized_series().floor().map(Column::from),
        Column::Scalar(s) => Ok(ScalarColumn::from_single_value_series(
            s.as_single_value_series().floor()?,
            s.len(),
        )
        .into()),
    }
}

pub(super) fn ceil(c: &Column) -> PolarsResult<Column> {
    match c {
        Column::Series(s) => s.ceil().map(Column::from),
        Column::Scalar(s) if s.is_empty() => s.as_materialized_series().ceil().map(Column::from),
        Column::Scalar(s) => Ok(ScalarColumn::from_single_value_series(
            s.as_single_value_series().ceil()?,
            s.len(),
        )
        .into()),
    }
}
