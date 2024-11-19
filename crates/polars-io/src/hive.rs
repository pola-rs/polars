use polars_core::frame::column::ScalarColumn;
use polars_core::frame::DataFrame;
use polars_core::prelude::Column;
use polars_core::series::Series;

/// Materializes hive partitions.
/// We have a special num_rows arg, as df can be empty when a projection contains
/// only hive partition columns.
///
/// The `hive_partition_columns` must be ordered by their position in the `reader_schema`
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

        let out_width: usize = df.width() + hive_columns.len();
        let df_columns = df.get_columns();
        let mut out_columns = Vec::with_capacity(out_width);

        // Merge `df_columns` and `hive_columns` such that the result columns are in the order
        // they appear in `reader_schema`. Note `reader_schema` may contain extra columns that were
        // excluded after a projection pushdown.

        // Safety: Both `df_columns` and `hive_columns` are non-empty.
        let mut series_arr = [df_columns, hive_columns.as_slice()];
        let mut schema_idx_arr = [
            // `unwrap_or(0)`: The first column could be a row_index column that doesn't exist in the `reader_schema`.
            reader_schema.index_of(series_arr[0][0].name()).unwrap_or(0),
            reader_schema.index_of(series_arr[1][0].name()).unwrap(),
        ];

        loop {
            // Take from the side whose next column appears earlier in the `reader_schema`.
            let arg_min = if schema_idx_arr[1] < schema_idx_arr[0] {
                1
            } else {
                0
            };

            out_columns.push(series_arr[arg_min][0].clone());
            series_arr[arg_min] = &series_arr[arg_min][1..];

            if series_arr[arg_min].is_empty() {
                break;
            }

            let Some(i) = reader_schema.index_of(series_arr[arg_min][0].name()) else {
                // All columns in `df_columns` should be present in `reader_schema` except for a row_index column.
                // We assume that if a row_index column exists it is always the first column and handle that at
                // initialization.
                debug_assert_eq!(arg_min, 1);
                break;
            };

            schema_idx_arr[arg_min] = i;
        }

        out_columns.extend_from_slice(series_arr[0]);
        out_columns.extend_from_slice(series_arr[1]);

        *df = unsafe { DataFrame::new_no_checks(num_rows, out_columns) };
    }
}
