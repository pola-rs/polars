use polars_core::error::PolarsResult;
use polars_core::prelude::Column;

pub(super) fn cum_count(s: &Column, reverse: bool) -> PolarsResult<Column> {
    // @scalar-opt
    polars_ops::prelude::cum_count(s.as_materialized_series(), reverse).map(Column::from)
}

pub(super) fn cum_sum(s: &Column, reverse: bool) -> PolarsResult<Column> {
    // @scalar-opt
    polars_ops::prelude::cum_sum(s.as_materialized_series(), reverse).map(Column::from)
}

pub(super) fn cum_prod(s: &Column, reverse: bool) -> PolarsResult<Column> {
    // @scalar-opt
    polars_ops::prelude::cum_prod(s.as_materialized_series(), reverse).map(Column::from)
}

pub(super) fn cum_min(s: &Column, reverse: bool) -> PolarsResult<Column> {
    // @scalar-opt
    polars_ops::prelude::cum_min(s.as_materialized_series(), reverse).map(Column::from)
}

pub(super) fn cum_max(s: &Column, reverse: bool) -> PolarsResult<Column> {
    // @scalar-opt
    polars_ops::prelude::cum_max(s.as_materialized_series(), reverse).map(Column::from)
}

pub(super) fn cum_mean(s: &Column, reverse: bool) -> PolarsResult<Column> {
    // @scalar-opt
    polars_ops::prelude::cum_mean(s.as_materialized_series(), reverse).map(Column::from)
}
