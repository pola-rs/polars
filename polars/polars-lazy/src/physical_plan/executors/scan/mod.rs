#[cfg(feature = "csv-file")]
mod csv;
#[cfg(feature = "ipc")]
mod ipc;
#[cfg(feature = "json")]
mod ndjson;
#[cfg(feature = "parquet")]
mod parquet;

use std::fs::File;
use std::mem;
use std::path::Path;

#[cfg(feature = "csv-file")]
pub(crate) use csv::CsvExec;
#[cfg(feature = "ipc")]
pub(crate) use ipc::IpcExec;
#[cfg(feature = "parquet")]
pub(crate) use parquet::ParquetExec;
use polars_io::aggregations::ScanAggregation;
use polars_io::csv::CsvEncoding;
use polars_io::prelude::*;

use super::*;
use crate::prelude::*;

#[cfg(any(feature = "ipc", feature = "parquet"))]
type Projection = Option<Vec<usize>>;
#[cfg(any(feature = "ipc", feature = "parquet"))]
type StopNRows = Option<usize>;
#[cfg(any(feature = "ipc", feature = "parquet"))]
type Aggregation<'a> = Option<&'a [ScanAggregation]>;
#[cfg(any(feature = "ipc", feature = "parquet"))]
type Predicate = Option<Arc<dyn PhysicalIoExpr>>;

#[cfg(any(feature = "ipc", feature = "parquet"))]
fn prepare_scan_args<'a>(
    path: &Path,
    predicate: &Option<Arc<dyn PhysicalExpr>>,
    with_columns: &mut Option<Arc<Vec<String>>>,
    schema: &mut SchemaRef,
    n_rows: Option<usize>,
    aggregate: &'a [ScanAggregation],
) -> (File, Projection, StopNRows, Aggregation<'a>, Predicate) {
    let file = std::fs::File::open(&path).unwrap();

    let with_columns = mem::take(with_columns);
    let schema = mem::take(schema);

    let projection: Option<Vec<_>> = with_columns.map(|with_columns| {
        with_columns
            .iter()
            .map(|name| schema.index_of(name).unwrap())
            .collect()
    });

    let n_rows = set_n_rows(n_rows);
    let aggregate = if aggregate.is_empty() {
        None
    } else {
        Some(aggregate)
    };
    let predicate = predicate
        .clone()
        .map(|expr| Arc::new(PhysicalIoHelper { expr }) as Arc<dyn PhysicalIoExpr>);

    (file, projection, n_rows, aggregate, predicate)
}

/// Producer of an in memory DataFrame
pub struct DataFrameExec {
    pub(crate) df: Arc<DataFrame>,
    pub(crate) selection: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) projection: Option<Arc<Vec<String>>>,
}

impl Executor for DataFrameExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let df = mem::take(&mut self.df);
        let mut df = Arc::try_unwrap(df).unwrap_or_else(|df| (*df).clone());

        // projection should be before selection as those are free
        // TODO: this is only the case if we don't create new columns
        if let Some(projection) = &self.projection {
            state.may_set_schema(&df, projection.len());
            df = df.select(projection.as_ref())?;
        }

        if let Some(selection) = &self.selection {
            let s = selection.evaluate(&df, state)?;
            let mask = s.bool().map_err(|_| {
                PolarsError::ComputeError("filter predicate was not of type boolean".into())
            })?;
            df = df.filter(mask)?;
        }
        state.clear_schema_cache();

        if let Some(limit) = set_n_rows(None) {
            Ok(df.head(Some(limit)))
        } else {
            Ok(df)
        }
    }
}

pub(crate) struct AnonymousScanExec {
    pub(crate) function: Arc<dyn AnonymousScan>,
    pub(crate) options: AnonymousScanOptions,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
}

impl Executor for AnonymousScanExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.record(
            || {
                let mut df = self.function.scan(self.options.clone())?;
                if let Some(predicate) = &self.predicate {
                    let s = predicate.evaluate(&df, state)?;
                    let mask = s.bool().map_err(|_| {
                        PolarsError::ComputeError("filter predicate was not of type boolean".into())
                    })?;
                    df = df.filter(mask)?;
                }
                Ok(df)
            },
            "anonymous_scan".into(),
        )
    }
}
