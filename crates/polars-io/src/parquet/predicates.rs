use arrow::io::parquet::read::statistics::{deserialize, Statistics};
use arrow::io::parquet::read::RowGroupMetaData;
use polars_core::prelude::*;

use crate::predicates::{BatchStats, ColumnStats, PhysicalIoExpr};
use crate::ArrowResult;

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
    md: &[RowGroupMetaData],
    arrow_schema: &ArrowSchema,
    rg: Option<usize>,
) -> ArrowResult<Option<BatchStats>> {
    let mut schema = Schema::with_capacity(arrow_schema.fields.len());
    let mut stats = vec![];

    for fld in &arrow_schema.fields {
        // note that we only select a single row group.
        let st = match rg {
            None => deserialize(fld, md)?,
            // we select a single row group and collect only those stats
            Some(rg) => deserialize(fld, &md[rg..rg + 1])?,
        };
        schema.with_column((&fld.name).into(), (&fld.data_type).into());
        stats.push(ColumnStats::from_arrow_stats(st, fld));
    }

    Ok(if stats.is_empty() {
        None
    } else {
        Some(BatchStats::new(schema, stats))
    })
}

pub(super) fn read_this_row_group(
    predicate: Option<&Arc<dyn PhysicalIoExpr>>,
    file_metadata: &arrow::io::parquet::read::FileMetaData,
    schema: &ArrowSchema,
    rg: usize,
) -> PolarsResult<bool> {
    if let Some(pred) = &predicate {
        if let Some(pred) = pred.as_stats_evaluator() {
            if let Some(stats) = collect_statistics(&file_metadata.row_groups, schema, Some(rg))? {
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
