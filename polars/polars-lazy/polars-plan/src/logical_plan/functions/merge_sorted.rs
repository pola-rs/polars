use polars_core::prelude::*;
use polars_ops::prelude::*;

pub(super) fn merge_sorted(df: &DataFrame, column: &str) -> PolarsResult<DataFrame> {
    // Safety:
    // the dtype is known
    let (left_cols, right_cols) = unsafe {
        (
            df.get_columns()
                .iter()
                .map(|s| {
                    Series::from_chunks_and_dtype_unchecked(
                        s.name(),
                        s.chunks()[..1].to_vec(),
                        s.dtype(),
                    )
                })
                .collect::<Vec<_>>(),
            df.get_columns()
                .iter()
                .map(|s| {
                    Series::from_chunks_and_dtype_unchecked(
                        s.name(),
                        s.chunks()[1..].to_vec(),
                        s.dtype(),
                    )
                })
                .collect::<Vec<_>>(),
        )
    };

    let left = DataFrame::new_no_checks(left_cols);
    let right = DataFrame::new_no_checks(right_cols);

    let lhs = left.column(column)?;
    let rhs = right.column(column)?;
    _merge_sorted_dfs(&left, &right, lhs, rhs, true)
}
