use polars_ops::series::index_of as index_of_op;

use super::*;

pub(super) fn index_of(s: &mut [Column]) -> PolarsResult<Option<Column>> {
    let Some(series) = s[0].as_series() else {
        return Ok(None);
    };
    let Some(value) = s[1].as_scalar_column().map(|sc| sc.scalar().value()) else {
        return Ok(None);
    };
    let result = index_of_op(series, value)?;
    Ok(result.map(|r| Column::new(series.name().clone(), [r as IdxSize])))
}
