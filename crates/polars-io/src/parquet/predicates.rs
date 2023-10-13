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
    md: &RowGroupMetaData,
    arrow_schema: &ArrowSchema,
) -> ArrowResult<Option<BatchStats>> {
    let mut schema = Schema::with_capacity(arrow_schema.fields.len());
    let mut stats = vec![];

    for fld in &arrow_schema.fields {
        let st = deserialize(fld, md)?;
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
    md: &RowGroupMetaData,
    schema: &ArrowSchema,
) -> PolarsResult<bool> {
    if let Some(pred) = &predicate {
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
