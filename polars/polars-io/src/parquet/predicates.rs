use crate::predicates::PhysicalIoExpr;
use crate::ArrowResult;
use arrow::array::Array;
use arrow::compute::concatenate::concatenate;
use arrow::io::parquet::read::statistics::{self, deserialize, Statistics};
use arrow::io::parquet::read::RowGroupMetaData;
use polars_core::prelude::*;

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
        match &self.0.null_count {
            statistics::Count::Single(arr) => {
                if arr.is_valid(0) {
                    Some(arr.value(0) as usize)
                } else {
                    None
                }
            }
            _ => None,
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
    pub fn get_stats(&self, column: &str) -> polars_core::error::Result<&ColumnStats> {
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
) -> ArrowResult<Option<BatchStats>> {
    let mut schema = Schema::with_capacity(arrow_schema.fields.len());
    let mut stats = vec![];

    for fld in &arrow_schema.fields {
        let st = deserialize(fld, md)?;
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
) -> Result<bool> {
    if let Some(pred) = &predicate {
        if let Some(pred) = pred.as_stats_evaluator() {
            if let Some(stats) = collect_statistics(&file_metadata.row_groups, schema)? {
                let should_read = pred.should_read(&stats);
                // a parquet file may not have statistics of all columns
                if matches!(should_read, Ok(false)) {
                    return Ok(false);
                } else if !matches!(should_read, Err(PolarsError::NotFound(_))) {
                    let _ = should_read?;
                }
            }
        }
    }
    Ok(true)
}
