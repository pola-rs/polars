use polars_core::frame::DataFrame;
use polars_core::frame::column::ScalarColumn;
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
) {
    let num_rows = df.height();

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

        let mut merged = Vec::with_capacity(df.width() + hive_columns.len());

        // `hive_partitions_from_paths()` guarantees `hive_columns` is sorted by their appearance in `reader_schema`.
        merge_sorted_to_schema_order(
            &mut unsafe { df.get_columns_mut().drain(..) },
            &mut hive_columns.into_iter(),
            reader_schema,
            &mut merged,
        );

        *df = unsafe { DataFrame::new_no_checks(num_rows, merged) };
    }
}

/// Merge 2 lists of columns into one, where each list contains columns ordered such that their indices
/// in the `schema` are in ascending order.
///
/// Layouts:
/// * `cols_lhs`: `[row_index?, ..schema_columns?, ..other_left?]`
///   * If the first item in `cols_lhs` is not found in the schema, it will be assumed to be a
///     `row_index` column and placed first into the result.
/// * `cols_rhs`: `[..schema_columns? ..other_right?]`
///
/// Output:
/// * `[..schema_columns?, ..other_left?, ..other_right?]`
///
/// Note: The `row_index` column should be handled before calling this function.
///
/// # Panics
/// Panics if either `cols_lhs` or `cols_rhs` is empty.
pub fn merge_sorted_to_schema_order<'a, D>(
    cols_lhs: &'a mut dyn Iterator<Item = Column>,
    cols_rhs: &'a mut dyn Iterator<Item = Column>,
    schema: &polars_schema::Schema<D>,
    output: &'a mut Vec<Column>,
) {
    merge_sorted_to_schema_order_impl(cols_lhs, cols_rhs, output, &|v| schema.index_of(v.name()))
}

pub fn merge_sorted_to_schema_order_impl<'a, T, O>(
    cols_lhs: &'a mut dyn Iterator<Item = T>,
    cols_rhs: &'a mut dyn Iterator<Item = T>,
    output: &mut O,
    get_opt_index: &dyn for<'b> Fn(&'b T) -> Option<usize>,
) where
    O: Extend<T>,
{
    let mut series_arr = [cols_lhs.peekable(), cols_rhs.peekable()];

    (|| {
        let (Some(a), Some(b)) = (
            series_arr[0]
                .peek()
                .and_then(|x| get_opt_index(x).or(Some(0))),
            series_arr[1].peek().and_then(get_opt_index),
        ) else {
            return;
        };

        let mut schema_idx_arr = [a, b];

        loop {
            // Take from the side whose next column appears earlier in the `schema`.
            let arg_min = if schema_idx_arr[1] < schema_idx_arr[0] {
                1
            } else {
                0
            };

            output.extend([series_arr[arg_min].next().unwrap()]);

            let Some(v) = series_arr[arg_min].peek() else {
                return;
            };

            let Some(i) = get_opt_index(v) else {
                // All columns in `cols_lhs` should be present in `schema` except for a row_index column.
                // We assume that if a row_index column exists it is always the first column and handle that at
                // initialization.
                debug_assert_eq!(arg_min, 1);
                break;
            };

            schema_idx_arr[arg_min] = i;
        }
    })();

    let [a, b] = series_arr;
    output.extend(a);
    output.extend(b);
}
