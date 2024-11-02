use polars_core::prelude::*;

pub fn as_struct(cols: &[Column]) -> PolarsResult<Column> {
    let Some(fst) = cols.first() else {
        polars_bail!(nyi = "turning no columns as_struct");
    };

    let mut min_length = usize::MAX;
    let mut max_length = usize::MIN;

    for col in cols {
        let len = col.len();

        min_length = min_length.min(len);
        max_length = max_length.max(len);
    }

    // @NOTE: Any additional errors should be handled by the StructChunked::from_columns
    let length = if min_length == 0 { 0 } else { max_length };

    Ok(StructChunked::from_columns(fst.name().clone(), length, cols)?.into_column())
}
