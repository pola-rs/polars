use polars_core::frame::DataFrame;
use polars_core::schema::IndexOfSchema;
use polars_core::series::Series;

/// Materializes hive partitions.
/// We have a special num_rows arg, as df can be empty when a projection contains
/// only hive partition columns.
///
/// # Safety
///
/// num_rows equals the height of the df when the df height is non-zero.
pub(crate) fn materialize_hive_partitions<S: IndexOfSchema>(
    df: &mut DataFrame,
    reader_schema: &S,
    hive_partition_columns: Option<&[Series]>,
    num_rows: usize,
) {
    if let Some(hive_columns) = hive_partition_columns {
        let Some(first) = hive_columns.first() else {
            return;
        };

        if reader_schema.index_of(first.name()).is_some() {
            // Insert these hive columns in the order they are stored in the file.
            for s in hive_columns {
                let i = match df.get_columns().binary_search_by_key(
                    &reader_schema.index_of(s.name()).unwrap_or(usize::MAX),
                    |s| reader_schema.index_of(s.name()).unwrap_or(usize::MIN),
                ) {
                    Ok(i) => i,
                    Err(i) => i,
                };

                df.insert_column(i, s.new_from_index(0, num_rows)).unwrap();
            }
        } else {
            for s in hive_columns {
                unsafe { df.with_column_unchecked(s.new_from_index(0, num_rows)) };
            }
        }
    }
}
