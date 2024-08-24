use polars_core::prelude::*;
use polars_parquet::read::statistics::{deserialize, Statistics};

use crate::parquet::read::metadata::PartitionedColumnChunkMD;
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

/// Collect the statistics in a row-group
pub(crate) fn collect_statistics(
    part_md: &PartitionedColumnChunkMD,
    schema: &ArrowSchema,
) -> PolarsResult<Option<BatchStats>> {
    // TODO! fix this performance. This is a full sequential scan.
    let stats = schema
        .fields
        .iter()
        .map(|field| match part_md.get_partitions(&field.name) {
            Some(md) => {
                let st = deserialize(field, &md)?;
                Ok(ColumnStats::from_arrow_stats(st, field))
            },
            None => Ok(ColumnStats::new(field.into(), None, None, None)),
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    if stats.is_empty() {
        return Ok(None);
    }

    Ok(Some(BatchStats::new(
        Arc::new(schema.into()),
        stats,
        Some(part_md.num_rows()),
    )))
}

pub fn read_this_row_group(
    predicate: Option<&dyn PhysicalIoExpr>,
    part_md: &PartitionedColumnChunkMD,
    schema: &ArrowSchema,
) -> PolarsResult<bool> {
    if let Some(pred) = predicate {
        if let Some(pred) = pred.as_stats_evaluator() {
            if let Some(stats) = collect_statistics(part_md, schema)? {
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
