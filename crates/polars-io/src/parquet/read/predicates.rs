use arrow::datatypes::ArrowSchemaRef;
use polars_core::prelude::*;
use polars_parquet::read::statistics::{deserialize, Statistics};
use polars_parquet::read::RowGroupMetaData;

use crate::predicates::{BatchStats, ColumnStats, PhysicalIoExpr};

impl ColumnStats {
    fn from_arrow_stats(stats: Statistics, field: &ArrowField) -> Self {
        Self::new(
            field.into(),
            Some(Series::try_from(("", stats.null_count)).unwrap()),
            Some(Series::try_from(("", stats.min_value)).unwrap()),
            Some(Series::try_from(("", stats.max_value)).unwrap()),
        )
    }
}

/// Collect the statistics in a column chunk.
pub(crate) fn collect_statistics(
    md: &RowGroupMetaData,
    schema: &ArrowSchema,
) -> PolarsResult<Option<BatchStats>> {
    let stats = schema
        .fields
        .iter()
        .map(|field| {
            let st = deserialize(field, md)?;
            Ok(ColumnStats::from_arrow_stats(st, field))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    if stats.is_empty() {
        return Ok(None);
    }

    Ok(Some(BatchStats::new(
        Arc::new(schema.into()),
        stats,
        Some(md.num_rows()),
    )))
}

pub(super) fn read_this_row_group(
    predicate: Option<&dyn PhysicalIoExpr>,
    md: &RowGroupMetaData,
    schema: &ArrowSchemaRef,
) -> PolarsResult<bool> {
    if let Some(pred) = predicate {
        if let Some(pred) = pred.as_stats_evaluator() {
            if let Some(stats) = collect_statistics(md, schema)? {
                let should_read = pred.should_read(&stats);
                // a parquet file may not have statistics of all columns
                if matches!(should_read, Ok(false)) {
                    return Ok(false);
                } else if !matches!(should_read, Err(PolarsError::ColumnNotFound(_))) {
                    let _ = should_read?;
                }
            }
        }
    }
    Ok(true)
}
