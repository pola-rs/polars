use crate::ArrowResult;
use arrow::io::parquet::read::statistics::{
    deserialize_statistics, PrimitiveStatistics, Statistics, Utf8Statistics,
};
use arrow::io::parquet::read::ColumnChunkMetaData;
use polars_core::prelude::*;

/// The statistics for a column in a Parquet file
/// they typically hold
/// - max value
/// - min value
/// - null_count
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct ColumnStats(Box<dyn Statistics>);

impl ColumnStats {
    pub fn dtype(&self) -> DataType {
        self.0.data_type().into()
    }

    pub fn to_min_max(&self) -> Option<Series> {
        let name = "";
        use DataType::*;
        let s = match self.dtype() {
            Float64 => {
                let stats = self
                    .0
                    .as_any()
                    .downcast_ref::<PrimitiveStatistics<f64>>()
                    .unwrap();
                Series::new(name, [stats.min_value, stats.max_value])
            }
            Float32 => {
                let stats = self
                    .0
                    .as_any()
                    .downcast_ref::<PrimitiveStatistics<f32>>()
                    .unwrap();
                Series::new(name, [stats.min_value, stats.max_value])
            }
            Int64 => {
                let stats = self
                    .0
                    .as_any()
                    .downcast_ref::<PrimitiveStatistics<i64>>()
                    .unwrap();
                Series::new(name, [stats.min_value, stats.max_value])
            }
            Int32 => {
                let stats = self
                    .0
                    .as_any()
                    .downcast_ref::<PrimitiveStatistics<i32>>()
                    .unwrap();
                Series::new(name, [stats.min_value, stats.max_value])
            }
            UInt32 => {
                let stats = self
                    .0
                    .as_any()
                    .downcast_ref::<PrimitiveStatistics<u32>>()
                    .unwrap();
                Series::new(name, [stats.min_value, stats.max_value])
            }
            UInt64 => {
                let stats = self
                    .0
                    .as_any()
                    .downcast_ref::<PrimitiveStatistics<u64>>()
                    .unwrap();
                Series::new(name, [stats.min_value, stats.max_value])
            }
            Utf8 => {
                let stats = self.0.as_any().downcast_ref::<Utf8Statistics>().unwrap();
                Series::new(
                    name,
                    [stats.min_value.as_deref(), stats.max_value.as_deref()],
                )
            }
            _ => return None,
        };
        Some(s)
    }
}

/// A collection of column stats with a known schema.
pub struct BatchStats {
    schema: Schema,
    stats: Vec<ColumnStats>,
}

impl BatchStats {
    pub fn get_stats(&self, column: &str) -> polars_core::error::Result<&ColumnStats> {
        self.schema.index_of(column).map(|i| &self.stats[i])
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}

/// Collect the statistics in a column chunk.
pub(crate) fn collect_statistics(
    md: &[ColumnChunkMetaData],
    schema: &ArrowSchema,
) -> ArrowResult<Option<BatchStats>> {
    let mut fields = vec![];
    let mut stats = vec![];

    for (column_chunk_md, fld) in md.iter().zip(&schema.fields) {
        if let Some(parquet_stats) = column_chunk_md.statistics() {
            let parquet_stats = parquet_stats?;
            let st = deserialize_statistics(&*parquet_stats)?;

            fields.push(fld.into());
            stats.push(ColumnStats(st));
        }
    }

    Ok(if fields.is_empty() {
        None
    } else {
        Some(BatchStats {
            schema: Schema::new(fields),
            stats,
        })
    })
}
