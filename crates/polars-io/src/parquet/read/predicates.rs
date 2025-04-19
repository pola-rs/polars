use polars_core::prelude::*;
use polars_parquet::read::RowGroupMetadata;
use polars_parquet::read::statistics::{ArrowColumnStatisticsArrays, deserialize_all};

/// Collect the statistics in a row-group
pub fn collect_statistics_with_live_columns(
    row_groups: &[RowGroupMetadata],
    schema: &ArrowSchema,
    live_columns: &PlIndexSet<PlSmallStr>,
) -> PolarsResult<Vec<Option<ArrowColumnStatisticsArrays>>> {
    if row_groups.is_empty() {
        return Ok((0..live_columns.len()).map(|_| None).collect());
    }

    let md = &row_groups[0];
    live_columns
        .iter()
        .map(|c| {
            let field = schema.get(c).unwrap();

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
