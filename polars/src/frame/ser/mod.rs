pub mod csv;
pub(crate) mod fork;
#[cfg(feature = "ipc")]
#[doc(cfg(feature = "ipc"))]
pub mod ipc;
#[cfg(feature = "json")]
#[doc(cfg(feature = "json"))]
pub mod json;
#[cfg(feature = "parquet")]
#[doc(cfg(feature = "parquet"))]
pub mod parquet;

#[cfg(feature = "lazy")]
use crate::lazy::prelude::PhysicalExpr;
use crate::prelude::*;
use crate::utils::accumulate_dataframes_vertical;
use arrow::array::ArrayRef;
use arrow::{
    csv::Reader as ArrowCsvReader, error::Result as ArrowResult, json::Reader as ArrowJsonReader,
    record_batch::RecordBatch,
};
use std::io::{Read, Seek, Write};
use std::sync::Arc;

#[cfg(not(feature = "lazy"))]
pub trait PhysicalExpr {
    fn evaluate(&self, df: &DataFrame) -> Result<Series>;
}

pub trait SerReader<R>
where
    R: Read + Seek,
{
    fn new(reader: R) -> Self;

    /// Rechunk to a single chunk after Reading file.
    fn set_rechunk(self, _rechunk: bool) -> Self
    where
        Self: std::marker::Sized,
    {
        self
    }

    /// Take the SerReader and return a parsed DataFrame.
    fn finish(self) -> Result<DataFrame>;
}

pub trait SerWriter<'a, W>
where
    W: Write,
{
    fn new(writer: &'a mut W) -> Self;
    fn finish(self, df: &mut DataFrame) -> Result<()>;
}

pub trait ArrowReader {
    fn next_record_batch(&mut self) -> ArrowResult<Option<RecordBatch>>;

    fn schema(&self) -> Arc<Schema>;
}

impl<R: Read> ArrowReader for ArrowCsvReader<R> {
    fn next_record_batch(&mut self) -> ArrowResult<Option<RecordBatch>> {
        self.next().map_or(Ok(None), |v| v.map(Some))
    }

    fn schema(&self) -> Arc<Schema> {
        self.schema()
    }
}

impl<R: Read> ArrowReader for ArrowJsonReader<R> {
    fn next_record_batch(&mut self) -> ArrowResult<Option<RecordBatch>> {
        self.next()
    }

    fn schema(&self) -> Arc<Schema> {
        self.schema()
    }
}

fn init_ca<T>(arr: &ArrayRef, field: &Field) -> ChunkedArray<T>
where
    T: PolarsDataType,
{
    ChunkedArray::new_from_chunks(field.name(), vec![arr.clone()])
}

fn arr_to_series(arr: &ArrayRef, field: &Field) -> Series {
    (field.name().as_str(), arr.clone()).into()
}

pub(crate) fn finish_reader<R: ArrowReader>(
    mut reader: R,
    rechunk: bool,
    stop_after_n_rows: Option<usize>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    aggregate: Option<&[ScanAggregation]>,
) -> Result<DataFrame> {
    let mut n_rows = 0;
    let mut parsed_dfs = Vec::with_capacity(1024);

    while let Some(batch) = reader.next_record_batch()? {
        n_rows += batch.num_rows();

        let columns = batch
            .columns()
            .iter()
            .zip(reader.schema().fields())
            .map(|(arr, field)| arr_to_series(arr, field))
            .collect();

        let mut df = DataFrame::new_no_checks(columns);

        if let Some(predicate) = &predicate {
            let s = predicate.evaluate(&df)?;
            let mask = s.bool().expect("filter predicates was not of type boolean");
            df = df.filter(mask)?;
        }

        if let Some(aggregate) = aggregate {
            let cols = aggregate
                .iter()
                .map(|scan_agg| scan_agg.evaluate_batch(&df).unwrap())
                .collect();
            if cfg!(debug_assertions) {
                df = DataFrame::new(cols).unwrap();
            } else {
                df = DataFrame::new_no_checks(cols)
            }
        }

        parsed_dfs.push(df);
        if let Some(n) = stop_after_n_rows {
            if n_rows >= n {
                break;
            }
        }
    }
    let mut df = accumulate_dataframes_vertical(parsed_dfs)?;

    if let Some(aggregate) = aggregate {
        let cols = aggregate
            .iter()
            .map(|scan_agg| scan_agg.finish(&df).unwrap())
            .collect();
        df = DataFrame::new_no_checks(cols)
    }

    match rechunk {
        true => Ok(df.agg_chunks()),
        false => Ok(df),
    }
}

pub(crate) enum ScanAggregation {
    Sum {
        column: String,
        alias: Option<String>,
    },
    Min {
        column: String,
        alias: Option<String>,
    },
    Max {
        column: String,
        alias: Option<String>,
    },
    First {
        column: String,
        alias: Option<String>,
    },
    Last {
        column: String,
        alias: Option<String>,
    },
}

impl ScanAggregation {
    /// Evaluate the aggregations per batch.
    pub(crate) fn evaluate_batch(&self, df: &DataFrame) -> Result<Series> {
        use ScanAggregation::*;
        let s = match self {
            Sum { column, .. } => df.column(column)?.sum_as_series(),
            Min { column, .. } => df.column(column)?.min_as_series(),
            Max { column, .. } => df.column(column)?.max_as_series(),
            First { column, .. } => df.column(column)?.head(Some(1)),
            Last { column, .. } => df.column(column)?.tail(Some(1)),
        };
        Ok(s)
    }

    /// After all batches are concatenated the aggregation is determined for the whole set.
    pub(crate) fn finish(&self, df: &DataFrame) -> Result<Series> {
        use ScanAggregation::*;
        match self {
            Sum { column, alias } => {
                let mut s = df.column(column)?.sum_as_series();
                if let Some(alias) = alias {
                    s.rename(alias);
                }
                Ok(s)
            }
            Min { column, alias } => {
                let mut s = df.column(column)?.min_as_series();
                if let Some(alias) = alias {
                    s.rename(alias);
                }
                Ok(s)
            }
            Max { column, alias } => {
                let mut s = df.column(column)?.max_as_series();
                if let Some(alias) = alias {
                    s.rename(alias);
                }
                Ok(s)
            }
            First { column, alias } => {
                let mut s = df.column(column)?.head(Some(1));
                if let Some(alias) = alias {
                    s.rename(alias);
                }
                Ok(s)
            }
            Last { column, alias } => {
                let mut s = df.column(column)?.tail(Some(1));
                if let Some(alias) = alias {
                    s.rename(alias);
                }
                Ok(s)
            }
        }
    }
}
