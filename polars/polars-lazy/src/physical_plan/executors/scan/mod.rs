#[cfg(feature = "csv-file")]
mod csv;
#[cfg(feature = "ipc")]
mod ipc;
#[cfg(feature = "json")]
mod ndjson;
#[cfg(feature = "parquet")]
mod parquet;

use std::mem;

#[cfg(feature = "csv-file")]
pub(crate) use csv::CsvExec;
#[cfg(feature = "ipc")]
pub(crate) use ipc::IpcExec;
#[cfg(feature = "parquet")]
pub(crate) use parquet::ParquetExec;
use polars_io::csv::CsvEncoding;
use polars_io::prelude::*;
use polars_plan::global::_set_n_rows_for_scan;
#[cfg(any(
    feature = "parquet",
    feature = "csv-file",
    feature = "ipc",
    feature = "cse"
))]
use polars_plan::logical_plan::FileFingerPrint;

use super::*;
use crate::prelude::*;

#[cfg(any(feature = "ipc", feature = "parquet"))]
type Projection = Option<Vec<usize>>;
#[cfg(any(feature = "ipc", feature = "parquet"))]
type StopNRows = Option<usize>;
#[cfg(any(feature = "ipc", feature = "parquet"))]
type Predicate = Option<Arc<dyn PhysicalIoExpr>>;

#[cfg(any(feature = "ipc", feature = "parquet"))]
fn prepare_scan_args(
    path: &std::path::Path,
    predicate: &Option<Arc<dyn PhysicalExpr>>,
    with_columns: &mut Option<Arc<Vec<String>>>,
    schema: &mut SchemaRef,
    n_rows: Option<usize>,
) -> (std::fs::File, Projection, StopNRows, Predicate) {
    let file = std::fs::File::open(path).unwrap();

    let with_columns = mem::take(with_columns);
    let schema = mem::take(schema);

    let projection: Option<Vec<_>> = with_columns.map(|with_columns| {
        with_columns
            .iter()
            .map(|name| schema.index_of(name).unwrap())
            .collect()
    });

    let n_rows = _set_n_rows_for_scan(n_rows);
    let predicate = predicate
        .clone()
        .map(|expr| Arc::new(PhysicalIoHelper { expr }) as Arc<dyn PhysicalIoExpr>);

    (file, projection, n_rows, predicate)
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
            df = df.select(projection.as_ref())?;
        }

        if let Some(selection) = &self.selection {
            let s = selection.evaluate(&df, state)?;
            let mask = s.bool().map_err(|_| {
                PolarsError::ComputeError("filter predicate was not of type boolean".into())
            })?;
            df = df.filter(mask)?;
        }

        if let Some(limit) = _set_n_rows_for_scan(None) {
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
            || match (self.function.allows_predicate_pushdown(), &self.predicate) {
                (true, Some(predicate)) => {
                    self.options.predicate = predicate.as_expression().cloned();
                    self.function.scan(self.options.clone())
                }
                (false, Some(predicate)) => {
                    let mut df = self.function.scan(self.options.clone())?;
                    let s = predicate.evaluate(&df, state)?;
                    let mask = s.bool().map_err(|_| {
                        PolarsError::ComputeError("filter predicate was not of type boolean".into())
                    })?;
                    df = df.filter(mask)?;

                    Ok(df)
                }
                _ => self.function.scan(self.options.clone()),
            },
            "anonymous_scan".into(),
        )
    }
}
