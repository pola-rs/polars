use polars_ops::series::index_of as index_of_op;

use super::*;

pub(super) fn index_of(s: &mut [Column]) -> PolarsResult<Option<Column>> {
    let Some(series) = s[0].as_series() else {
        return Ok(None);
    };
    let Some(value) = s[1].as_scalar_column().map(|sc| sc.scalar().value()) else {
        return Ok(None);
    };
    let is_sorted_flag = series.is_sorted_flag();
    let result = match is_sorted_flag {
        IsSorted::Not => index_of_op(series, value)?,
        // If the Series is sorted, we can use an optimized binary search to
        // find the value.
        IsSorted::Ascending | IsSorted::Descending => {
            let Ok(value) = s[1].as_materialized_series().strict_cast(series.dtype()) else {
                // If we can't cast, means we couldn't find the value.
                return Ok(None);
            };
            search_sorted(
                series,
                &value,
                SearchSortedSide::Any, // TODO should this be someting different?
                IsSorted::Descending == is_sorted_flag,
            )?
            .get(0)
            .map(|v| v as usize)
        },
    };
    Ok(result.map(|r| Column::new(series.name().clone(), [r as IdxSize])))
}
