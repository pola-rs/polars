use polars_core::prelude::Column;
use polars_error::{PolarsResult, polars_bail};
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

/// Returns Err if row index would overflow.
pub fn materialize_row_index_checked(
    name: PlSmallStr,
    offset: IdxSize,
    length: usize,
) -> PolarsResult<Column> {
    let length = IdxSize::try_from(length).unwrap();

    if length > (IdxSize::MAX - offset) {
        polars_bail!(
            ComputeError:
            "row index with offset {} overflows on dataframe with height {}",
            offset, length
        )
    }

    Ok(Column::new_idxsize_range(name, offset..offset + length))
}
