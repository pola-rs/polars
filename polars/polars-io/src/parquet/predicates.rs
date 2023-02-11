use arrow::compute::concatenate::concatenate;
use arrow::io::parquet::read::statistics::{deserialize, Statistics};
use arrow::io::parquet::read::RowGroupMetaData;
use polars_core::prelude::*;

use crate::predicates::PhysicalIoExpr;
use crate::ArrowResult;

/// The statistics for a column in a Parquet file
/// they typically hold
/// - max value
/// - min value
/// - null_count
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct ColumnStats(Statistics, Field);

impl ColumnStats {
    pub fn dtype(&self) -> DataType {
        self.1.data_type().clone()
    }

    pub fn null_count(&self) -> Option<usize> {
        match self.1.data_type() {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => None,
            _ => {
                // the array holds the null count for every row group
                // so we sum them to get them of the whole file.
                Series::try_from(("", self.0.null_count.clone()))
                    .unwrap()
                    .sum()
            }
        }
    }

    pub fn to_min_max(&self) -> Option<Series> {
        let max_val = &*self.0.max_value;
        let min_val = &*self.0.min_value;

        let dtype = DataType::from(min_val.data_type());
        if dtype.is_numeric() || matches!(dtype, DataType::Utf8) {
            let arr = concatenate(&[min_val, max_val]).unwrap();
            let s = Series::try_from(("", arr)).unwrap();
            if s.null_count() > 0 {
                None
            } else {
                Some(s)
            }
        } else {
            None
        }
    }
}

/// A collection of column stats with a known schema.
pub struct BatchStats {
    schema: Schema,
    stats: Vec<ColumnStats>,
}

impl BatchStats {
    pub fn get_stats(&self, column: &str) -> polars_core::error::PolarsResult<&ColumnStats> {
        self.schema.try_index_of(column).map(|i| &self.stats[i])
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
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
        schema.with_column(fld.name.to_string(), (&fld.data_type).into());
        stats.push(ColumnStats(st, Field::from(fld)));
    }

    Ok(if stats.is_empty() {
        None
    } else {
        Some(BatchStats { schema, stats })
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
