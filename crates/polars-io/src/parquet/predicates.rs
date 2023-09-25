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
pub struct ColumnStats {
    field: Field,
    // The array may hold the null count for every row group,
    // or for a single row group.
    null_count: Option<Series>,
    min_value: Option<Series>,
    max_value: Option<Series>,
}

impl ColumnStats {
    fn from_arrow_stats(stats: Statistics, field: &ArrowField) -> Self {
        Self {
            field: field.into(),
            null_count: Some(Series::try_from(("", stats.null_count)).unwrap()),
            min_value: Some(Series::try_from(("", stats.min_value)).unwrap()),
            max_value: Some(Series::try_from(("", stats.max_value)).unwrap()),
        }
    }

    pub fn dtype(&self) -> &DataType {
        self.field.data_type()
    }

    pub fn null_count(&self) -> Option<usize> {
        match self.field.data_type() {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => None,
            _ => {
                let s = self.null_count.as_ref()?;
                // if all null, there are no statistics.
                if s.null_count() != s.len() {
                    s.sum()
                } else {
                    None
                }
            },
        }
    }

    pub fn to_min_max(&self) -> Option<Series> {
        let max_val = self.max_value.as_ref()?;
        let min_val = self.min_value.as_ref()?;

        let dtype = min_val.dtype();

        if Self::use_min_max(dtype) {
            let mut min_max_values = min_val.clone();
            min_max_values.append(max_val).unwrap();
            if min_max_values.null_count() > 0 {
                None
            } else {
                Some(min_max_values)
            }
        } else {
            None
        }
    }

    pub fn to_min(&self) -> Option<&Series> {
        let min_val = self.min_value.as_ref()?;
        let dtype = min_val.dtype();

        if !Self::use_min_max(dtype) || min_val.len() != 1 {
            return None;
        }

        if min_val.null_count() > 0 {
            None
        } else {
            Some(min_val)
        }
    }

    pub fn to_max(&self) -> Option<&Series> {
        let max_val = self.max_value.as_ref()?;
        let dtype = max_val.dtype();

        if !Self::use_min_max(dtype) || max_val.len() != 1 {
            return None;
        }

        if max_val.null_count() > 0 {
            None
        } else {
            Some(max_val)
        }
    }

    fn use_min_max(dtype: &DataType) -> bool {
        dtype.is_numeric() || matches!(dtype, DataType::Utf8 | DataType::Binary)
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
        schema.with_column((&fld.name).into(), (&fld.data_type).into());
        stats.push(ColumnStats::from_arrow_stats(st, fld));
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
