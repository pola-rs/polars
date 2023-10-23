#[cfg(feature = "csv")]
mod csv;
#[cfg(feature = "ipc")]
mod ipc;
#[cfg(feature = "json")]
mod ndjson;
#[cfg(feature = "parquet")]
mod parquet;

use std::mem;
use std::ops::Deref;

#[cfg(feature = "csv")]
pub(crate) use csv::CsvExec;
#[cfg(feature = "ipc")]
pub(crate) use ipc::IpcExec;
#[cfg(feature = "parquet")]
pub(crate) use parquet::ParquetExec;
#[cfg(any(feature = "ipc", feature = "parquet"))]
use polars_io::predicates::PhysicalIoExpr;
use polars_io::prelude::*;
use polars_plan::global::_set_n_rows_for_scan;
#[cfg(any(feature = "parquet", feature = "csv", feature = "ipc", feature = "cse"))]
use polars_plan::logical_plan::FileFingerPrint;

use super::*;
use crate::physical_plan::expressions::phys_expr_to_io_expr;
use crate::prelude::*;

#[cfg(any(feature = "ipc", feature = "parquet"))]
type Projection = Option<Vec<usize>>;
#[cfg(any(feature = "ipc", feature = "parquet"))]
type Predicate = Option<Arc<dyn PhysicalIoExpr>>;

#[cfg(any(feature = "ipc", feature = "parquet"))]
fn prepare_scan_args(
    path: &std::path::Path,
    predicate: &Option<Arc<dyn PhysicalExpr>>,
    with_columns: &mut Option<Arc<Vec<String>>>,
    schema: &mut SchemaRef,
    has_row_count: bool,
    hive_partitions: Option<&[Series]>,
) -> (std::io::Result<std::fs::File>, Projection, Predicate) {
    let file = std::fs::File::open(path);

    let with_columns = mem::take(with_columns);
    let schema = mem::take(schema);

    let projection = materialize_projection(
        with_columns.as_deref().map(|cols| cols.deref()),
        &schema,
        hive_partitions,
        has_row_count,
    );

    let predicate = predicate.clone().map(phys_expr_to_io_expr);

    (file, projection, predicate)
}

/// Producer of an in memory DataFrame
pub struct DataFrameExec {
    pub(crate) df: Arc<DataFrame>,
    pub(crate) selection: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) projection: Option<Arc<Vec<String>>>,
    pub(crate) predicate_has_windows: bool,
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
            if self.predicate_has_windows {
                state.insert_has_window_function_flag()
            }
            let s = selection.evaluate(&df, state)?;
            if self.predicate_has_windows {
                state.clear_window_expr_cache()
            }
            let mask = s.bool().map_err(
                |_| polars_err!(ComputeError: "filter predicate was not of type boolean"),
            )?;
            df = df.filter(mask)?;
        }

        Ok(match _set_n_rows_for_scan(None) {
            Some(limit) => df.head(Some(limit)),
            None => df,
        })
    }
}

pub(crate) struct AnonymousScanExec {
    pub(crate) function: Arc<dyn AnonymousScan>,
    pub(crate) file_options: FileScanOptions,
    pub(crate) file_info: FileInfo,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) output_schema: Option<SchemaRef>,
    pub(crate) predicate_has_windows: bool,
}

impl Executor for AnonymousScanExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let mut args = AnonymousScanArgs {
            n_rows: self.file_options.n_rows,
            with_columns: self.file_options.with_columns.clone(),
            schema: self.file_info.schema.clone(),
            output_schema: self.output_schema.clone(),
            predicate: None,
        };
        if self.predicate.is_some() {
            state.insert_has_window_function_flag()
        }

        match (self.function.allows_predicate_pushdown(), &self.predicate) {
            (true, Some(predicate)) => state.record(
                || {
                    args.predicate = predicate.as_expression().cloned();
                    self.function.scan(args)
                },
                "anonymous_scan".into(),
            ),
            (false, Some(predicate)) => state.record(
                || {
                    let mut df = self.function.scan(args)?;
                    let s = predicate.evaluate(&df, state)?;
                    if self.predicate_has_windows {
                        state.clear_window_expr_cache()
                    }
                    let mask = s.bool().map_err(
                        |_| polars_err!(ComputeError: "filter predicate was not of type boolean"),
                    )?;
                    df = df.filter(mask)?;

                    Ok(df)
                },
                "anonymous_scan".into(),
            ),
            _ => state.record(|| self.function.scan(args), "anonymous_scan".into()),
        }
    }
}
