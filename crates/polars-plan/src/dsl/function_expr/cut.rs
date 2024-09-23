use polars_core::prelude::*;

pub(crate) fn cut(
    s: &Column,
    breaks: Vec<f64>,
    labels: Option<Vec<PlSmallStr>>,
    left_closed: bool,
    include_breaks: bool,
) -> PolarsResult<Column> {
    polars_ops::prelude::cut(
        s.as_materialized_series(),
        breaks,
        labels,
        left_closed,
        include_breaks,
    )
    .map(Column::from)
}

pub(crate) fn qcut(
    s: &Column,
    probs: Vec<f64>,
    labels: Option<Vec<PlSmallStr>>,
    left_closed: bool,
    allow_duplicates: bool,
    include_breaks: bool,
) -> PolarsResult<Column> {
    polars_ops::prelude::qcut(
        s.as_materialized_series(),
        probs,
        labels,
        left_closed,
        allow_duplicates,
        include_breaks,
    )
    .map(Column::from)
}
