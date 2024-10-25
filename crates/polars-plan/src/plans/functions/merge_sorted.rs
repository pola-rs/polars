use polars_core::prelude::*;
use polars_ops::prelude::*;

pub(super) fn merge_sorted(df: &DataFrame, column: &str) -> PolarsResult<DataFrame> {
    // SAFETY:
    // the dtype is known
    let (left_cols, right_cols) = unsafe {
        (
            df.get_columns()
                .iter()
                .map(|s| {
                    Series::from_chunks_and_dtype_unchecked(
                        s.name().clone(),
                        s.as_materialized_series().chunks()[..1].to_vec(),
                        s.dtype(),
                    )
                    .into()
                })
                .collect::<Vec<_>>(),
            df.get_columns()
                .iter()
                .map(|s| {
                    Series::from_chunks_and_dtype_unchecked(
                        s.name().clone(),
                        s.as_materialized_series().chunks()[1..].to_vec(),
                        s.dtype(),
                    )
                    .into()
                })
                .collect::<Vec<_>>(),
        )
    };

    let left = unsafe { DataFrame::new_no_checks_height_from_first(left_cols) };
    let right = unsafe { DataFrame::new_no_checks_height_from_first(right_cols) };

    let lhs = left.column(column)?;
    let rhs = right.column(column)?;
    _merge_sorted_dfs(
        &left,
        &right,
        lhs.as_materialized_series(),
        rhs.as_materialized_series(),
        true,
    )
}
