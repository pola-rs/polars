use arrow::array::{MutablePrimitiveArray, PrimitiveArray};
use arrow::pushable::Pushable;
use polars_core::prelude::*;
use polars_parquet::read::RowGroupMetadata;
use polars_parquet::read::statistics::{ArrowColumnStatisticsArrays, deserialize_all};

/// Collect the statistics in a row-group
pub fn collect_statistics_with_live_columns(
    row_groups: &[RowGroupMetadata],
    schema: &ArrowSchema,
    live_columns: &PlIndexSet<PlSmallStr>,
    row_index: Option<(&PlSmallStr, IdxSize)>,
) -> PolarsResult<Vec<Option<ArrowColumnStatisticsArrays>>> {
    if row_groups.is_empty() {
        return Ok((0..live_columns.len()).map(|_| None).collect());
    }

    let md = &row_groups[0];

    live_columns
        .iter()
        .map(|c| {
            let Some(field) = schema.get(c) else {
                // Should be the row index column

                let Some((name, mut offset)) = row_index else {
                    if cfg!(debug_assertions) {
                        panic!()
                    }
                    return Ok(None);
                };

                if c != name {
                    if cfg!(debug_assertions) {
                        panic!()
                    }
                    return Ok(None);
                }

                let null_count =
                    PrimitiveArray::<IdxSize>::full(row_groups.len(), 0, ArrowDataType::IDX_DTYPE);

                let mut distinct_count =
                    MutablePrimitiveArray::<IdxSize>::with_capacity(row_groups.len());
                let mut min_value =
                    MutablePrimitiveArray::<IdxSize>::with_capacity(row_groups.len());
                let mut max_value =
                    MutablePrimitiveArray::<IdxSize>::with_capacity(row_groups.len());

                for rg in row_groups.iter() {
                    let n_rows = IdxSize::try_from(rg.num_rows()).unwrap_or(IdxSize::MAX);
                    distinct_count.push_value(n_rows);

                    if offset.checked_add(n_rows).is_none() {
                        min_value.push_null();
                        max_value.push_null();
                        continue;
                    }

                    if n_rows == 0 {
                        min_value.push_null();
                        max_value.push_null();
                    } else {
                        min_value.push_value(offset);
                        max_value.push_value(offset + n_rows - 1);
                    }

                    offset = offset.saturating_add(n_rows);
                }

                let out = ArrowColumnStatisticsArrays {
                    null_count,
                    distinct_count: distinct_count.freeze(),
                    min_value: min_value.freeze().boxed(),
                    max_value: max_value.freeze().boxed(),
                };

                return Ok(Some(out));
            };

            // This can be None in the allow_missing_columns case.
            let Some(idxs) = md.columns_idxs_under_root_iter(&field.name) else {
                return Ok(None);
            };

            // 0 is possible for possible for empty structs.
            //
            // 2+ is for structs. We don't support reading nested statistics for now. It does not
            // really make any sense at the moment with how we structure statistics.
            if idxs.is_empty() || idxs.len() > 1 {
                return Ok(None);
            }

            let idx = idxs[0];
            Ok(deserialize_all(field, row_groups, idx)?)
        })
        .collect::<PolarsResult<Vec<_>>>()
}
