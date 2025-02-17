use super::*;

pub(super) fn clip(s: &[Column], has_min: bool, has_max: bool) -> PolarsResult<Column> {
    match (has_min, has_max) {
        (true, true) => polars_ops::series::clip(
            s[0].as_materialized_series(),
            s[1].as_materialized_series(),
            s[2].as_materialized_series(),
        ),
        (true, false) => polars_ops::series::clip_min(
            s[0].as_materialized_series(),
            s[1].as_materialized_series(),
        ),
        (false, true) => polars_ops::series::clip_max(
            s[0].as_materialized_series(),
            s[1].as_materialized_series(),
        ),
        _ => unreachable!(),
    }
    .map(Column::from)
}
