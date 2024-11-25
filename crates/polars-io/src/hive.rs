use polars_core::frame::column::ScalarColumn;
use polars_core::frame::DataFrame;
use polars_core::prelude::Column;
use polars_core::series::Series;

/// Materializes hive partitions.
/// We have a special num_rows arg, as df can be empty when a projection contains
/// only hive partition columns.
///
/// The `hive_partition_columns` must be ordered by their position in the `reader_schema`. The
/// columns will be materialized by their positions in the file schema if they exist, or otherwise
/// at the end.
///
/// # Safety
///
/// num_rows equals the height of the df when the df height is non-zero.
pub(crate) fn materialize_hive_partitions<D>(
    df: &mut DataFrame,
    reader_schema: &polars_schema::Schema<D>,
    hive_partition_columns: Option<&[Series]>,
    num_rows: usize,
) {
    if let Some(hive_columns) = hive_partition_columns {
        // Insert these hive columns in the order they are stored in the file.
        if hive_columns.is_empty() {
            return;
        }

        let hive_columns = hive_columns
            .iter()
            .map(|s| ScalarColumn::new(s.name().clone(), s.first(), num_rows).into())
            .collect::<Vec<Column>>();

        if reader_schema.index_of(hive_columns[0].name()).is_none() || df.width() == 0 {
            // Fast-path - all hive columns are at the end
            if df.width() == 0 {
                unsafe { df.set_height(num_rows) };
            }
            unsafe { df.hstack_mut_unchecked(&hive_columns) };
            return;
        }

        let df_columns = df.get_columns();
        let mut merged = Vec::with_capacity(df_columns.len() + hive_columns.len());

        // `hive_partitions_from_paths()` guarantees `hive_columns` is sorted by their appearance in `reader_schema`.
        merge_sorted_to_schema_order(
            df_columns,
            hive_columns.as_slice(),
            reader_schema,
            &mut merged,
        );

        *df = unsafe { DataFrame::new_no_checks(num_rows, merged) };
    }
}

/// Merge 2 sets of columns into one, where each set contains columns ordered such that their indices
/// in the `schema` are in ascending order.
///
/// Layouts:
/// * `df_columns`: `[row_index?, ..schema_columns]`
///   * `df_columns` must start with either a row_index column, or a schema column. This is important
///     as we assume that the first column in `df_columns` is a row_index column if it doesn't exist
///     in the `schema`.
/// * `hive_columns`: `[..schema_columns?, ..hive_columns?]`
///
/// # Panics
/// Panics if either `df_columns` or `hive_columns` is empty.
pub(crate) fn merge_sorted_to_schema_order<D>(
    df_columns: &[Column],
    hive_columns: &[Column],
    schema: &polars_schema::Schema<D>,
    output: &mut Vec<Column>,
) {
    // Safety: Both `df_columns` and `hive_columns` are non-empty.
    let mut series_arr = [df_columns, hive_columns];

    if let Some(i) = schema.index_of(series_arr[1][0].name()) {
        let mut schema_idx_arr = [
            // `unwrap_or(0)`: The first column could be a row_index column that doesn't exist in the `schema`.
            schema.index_of(series_arr[0][0].name()).unwrap_or(0),
            i,
        ];

        loop {
            // Take from the side whose next column appears earlier in the `schema`.
            let arg_min = if schema_idx_arr[1] < schema_idx_arr[0] {
                1
            } else {
                0
            };

            output.push(series_arr[arg_min][0].clone());
            series_arr[arg_min] = &series_arr[arg_min][1..];

            if series_arr[arg_min].is_empty() {
                break;
            }

            let Some(i) = schema.index_of(series_arr[arg_min][0].name()) else {
                // All columns in `df_columns` should be present in `schema` except for a row_index column.
                // We assume that if a row_index column exists it is always the first column and handle that at
                // initialization.
                debug_assert_eq!(arg_min, 1);
                break;
            };

            schema_idx_arr[arg_min] = i;
        }
    }

    output.extend_from_slice(series_arr[0]);
    output.extend_from_slice(series_arr[1]);
}
