use polars_ops::series::index_of as index_of_op;
use super::*;

pub(super) fn index_of(s: &mut [Column]) -> PolarsResult<Option<Column>> {
    let series = s[0].as_materialized_series();
    let value = s[1].as_materialized_series();
    if value.len() != 1 {
        polars_bail!(
            ComputeError:
            "there can only be a single value searched for in `index_of` expressions, but {} values were give",
            value.len(),
        );
    }
    let result = index_of_op(series, value)?;
    Ok(result.map(|r| Column::new(series.name().clone(), [r as IdxSize])))
}
