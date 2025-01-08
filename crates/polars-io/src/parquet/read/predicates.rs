use std::borrow::Cow;

use polars_core::config;
use polars_core::prelude::*;
use polars_parquet::read::statistics::{deserialize, Statistics};
use polars_parquet::read::RowGroupMetadata;

use crate::predicates::{BatchStats, ColumnStats, PhysicalIoExpr};

impl ColumnStats {
    fn from_arrow_stats(stats: Statistics, field: &ArrowField) -> Self {
        Self::new(
            field.into(),
            Some(Series::try_from((PlSmallStr::EMPTY, stats.null_count)).unwrap()),
            Some(Series::try_from((PlSmallStr::EMPTY, stats.min_value)).unwrap()),
            Some(Series::try_from((PlSmallStr::EMPTY, stats.max_value)).unwrap()),
        )
    }
}

/// Collect the statistics in a row-group
pub(crate) fn collect_statistics<'a>(
    md: &RowGroupMetadata,
    schema: &ArrowSchema,
    live_schema: &'a Schema,
) -> PolarsResult<Option<BatchStats<'a>>> {
    // TODO! fix this performance. This is a full sequential scan.
    let stats = live_schema
        .iter_fields()
        .map(|field| {
            if field.dtype().is_nested() {
                return Ok(ColumnStats::new(field.clone(), None, None, None));
            }

            let arrow_field = schema.get(&field.name).unwrap();
            let iter = md.columns_under_root_iter(&field.name).unwrap();

            Ok(ColumnStats::from_arrow_stats(
                deserialize(arrow_field, iter)?,
                arrow_field,
            ))
        })
        .collect::<PolarsResult<Cow<_>>>()?;

    if stats.is_empty() {
        return Ok(None);
    }

    Ok(Some(BatchStats::new(
        live_schema,
        stats,
        Some(md.num_rows()),
    )))
}

pub fn read_this_row_group(
    predicate: Option<&dyn PhysicalIoExpr>,
    md: &RowGroupMetadata,
    schema: &ArrowSchema,
    live_schema: &Schema,
) -> PolarsResult<bool> {
    if std::env::var("POLARS_NO_PARQUET_STATISTICS").is_ok() {
        return Ok(true);
    }

    let mut should_read = true;

    if let Some(pred) = predicate {
        if let Some(pred) = pred.as_stats_evaluator() {
            if let Some(stats) = collect_statistics(md, schema, live_schema)? {
                let pred_result = pred.should_read(&stats);

                // a parquet file may not have statistics of all columns
                match pred_result {
                    Err(PolarsError::ColumnNotFound(errstr)) => {
                        return Err(PolarsError::ColumnNotFound(errstr))
                    },
                    Ok(false) => should_read = false,
                    _ => {},
                }
            }
        }

        if config::verbose() {
            if should_read {
                eprintln!(
                    "parquet row group must be read, statistics not sufficient for predicate."
                );
            } else {
                eprintln!("parquet row group can be skipped, the statistics were sufficient to apply the predicate.");
            }
        }
    }

    Ok(should_read)
}
