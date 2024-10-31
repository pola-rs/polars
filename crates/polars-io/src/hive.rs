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

        let hive_columns_sc = hive_columns
            .iter()
            .map(|s| ScalarColumn::new(s.name().clone(), s.first(), num_rows).into())
            .collect::<Vec<Column>>();

        if reader_schema.index_of(hive_columns[0].name()).is_none() || df.width() == 0 {
            // Fast-path - all hive columns are at the end
            if df.width() == 0 {
                unsafe { df.set_height(num_rows) };
            }
            unsafe { df.hstack_mut_unchecked(&hive_columns_sc) };
            return;
        }

        let out_width: usize = df.width() + hive_columns.len();
        let df_columns = df.get_columns();
        let mut out_columns = Vec::with_capacity(out_width);

        // We have a slightly involved algorithm here because `reader_schema` may contain extra
        // columns that were excluded from a projection pushdown.

        // Safety: These are both non-empty at the start
        let mut series_arr = [df_columns, hive_columns_sc.as_slice()];
        let mut schema_idx_arr = [
            reader_schema.index_of(series_arr[0][0].name()).unwrap(),
            reader_schema.index_of(series_arr[1][0].name()).unwrap(),
        ];

        loop {
            let arg_min = if schema_idx_arr[0] < schema_idx_arr[1] {
                0
            } else {
                1
            };

            out_columns.push(series_arr[arg_min][0].clone());
            series_arr[arg_min] = &series_arr[arg_min][1..];

            if series_arr[arg_min].is_empty() {
                break;
            }

            let Some(i) = reader_schema.index_of(series_arr[arg_min][0].name()) else {
                break;
            };

            schema_idx_arr[arg_min] = i;
        }

        out_columns.extend_from_slice(series_arr[0]);
        out_columns.extend_from_slice(series_arr[1]);

        *df = unsafe { DataFrame::new_no_checks(num_rows, out_columns) };
    }
}
