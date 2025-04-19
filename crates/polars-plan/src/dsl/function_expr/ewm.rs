use super::*;

pub(super) fn ewm_mean(s: &Column, options: EWMOptions) -> PolarsResult<Column> {
    polars_ops::prelude::ewm_mean(s.as_materialized_series(), options).map(Column::from)
}

pub(super) fn ewm_std(s: &Column, options: EWMOptions) -> PolarsResult<Column> {
    polars_ops::prelude::ewm_std(s.as_materialized_series(), options).map(Column::from)
}

pub(super) fn ewm_var(s: &Column, options: EWMOptions) -> PolarsResult<Column> {
    polars_ops::prelude::ewm_var(s.as_materialized_series(), options).map(Column::from)
}
