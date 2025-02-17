use super::*;

pub(super) fn abs(s: &Column) -> PolarsResult<Column> {
    polars_ops::prelude::abs(s.as_materialized_series()).map(Column::from)
}
