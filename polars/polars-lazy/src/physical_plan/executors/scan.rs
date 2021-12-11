use super::*;
use crate::logical_plan::CsvParserOptions;
#[cfg(feature = "ipc")]
use crate::logical_plan::IpcOptions;
use crate::utils::try_path_to_str;
use polars_io::prelude::*;
use polars_io::{csv::CsvEncoding, ScanAggregation};
#[cfg(any(feature = "ipc", feature = "parquet"))]
use std::fs::File;
use std::mem;
use std::path::Path;

fn cache_hit(
    path: &Path,
    predicate: &Option<Arc<dyn PhysicalExpr>>,
    state: &ExecutionState,
) -> (String, Option<DataFrame>) {
    let path_str = try_path_to_str(path).unwrap();
    let cache_key = match predicate {
        Some(predicate) => format!("{}{:?}", path_str, predicate.as_expression()),
        None => path_str.to_string(),
    };
    let cached = state.cache_hit(&cache_key);
    (cache_key, cached)
}

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
    with_columns: &mut Option<Vec<String>>,
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
            .map(|name| schema.column_with_name(name).unwrap().0)
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

#[cfg(feature = "ipc")]
pub struct IpcExec {
    pub(crate) path: PathBuf,
    pub(crate) schema: SchemaRef,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) aggregate: Vec<ScanAggregation>,
    pub(crate) options: IpcOptions,
}

#[cfg(feature = "ipc")]
impl Executor for IpcExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let (cache_key, cached) = cache_hit(&self.path, &self.predicate, state);
        if let Some(df) = cached {
            return Ok(df);
        }
        let (file, projection, n_rows, aggregate, predicate) = prepare_scan_args(
            &self.path,
            &self.predicate,
            &mut self.options.with_columns,
            &mut self.schema,
            self.options.n_rows,
            &self.aggregate,
        );
        let df = IpcReader::new(file)
            .with_n_rows(n_rows)
            .finish_with_scan_ops(
                predicate,
                aggregate,
                projection.as_ref().map(|v| v.as_ref()),
            )?;

        if self.options.cache {
            state.store_cache(cache_key, df.clone())
        }
        if state.verbose {
            println!("ipc {:?} read", self.path);
        }

        Ok(df)
    }
}

#[cfg(feature = "parquet")]
pub struct ParquetExec {
    path: PathBuf,
    schema: SchemaRef,
    with_columns: Option<Vec<String>>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    aggregate: Vec<ScanAggregation>,
    n_rows: Option<usize>,
    cache: bool,
}

#[cfg(feature = "parquet")]
impl ParquetExec {
    pub(crate) fn new(
        path: PathBuf,
        schema: SchemaRef,
        with_columns: Option<Vec<String>>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        aggregate: Vec<ScanAggregation>,
        n_rows: Option<usize>,
        cache: bool,
    ) -> Self {
        ParquetExec {
            path,
            schema,
            with_columns,
            predicate,
            aggregate,
            n_rows,
            cache,
        }
    }
}

#[cfg(feature = "parquet")]
impl Executor for ParquetExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let (cache_key, cached) = cache_hit(&self.path, &self.predicate, state);
        if let Some(df) = cached {
            return Ok(df);
        }
        let (file, projection, n_rows, aggregate, predicate) = prepare_scan_args(
            &self.path,
            &self.predicate,
            &mut self.with_columns,
            &mut self.schema,
            self.n_rows,
            &self.aggregate,
        );

        let df = ParquetReader::new(file)
            .with_n_rows(n_rows)
            .finish_with_scan_ops(
                predicate,
                aggregate,
                projection.as_ref().map(|v| v.as_ref()),
            )?;

        if self.cache {
            state.store_cache(cache_key, df.clone())
        }
        if state.verbose {
            println!("parquet {:?} read", self.path);
        }

        Ok(df)
    }
}

#[cfg(feature = "csv-file")]
pub struct CsvExec {
    pub path: PathBuf,
    pub schema: SchemaRef,
    pub options: CsvParserOptions,
    pub predicate: Option<Arc<dyn PhysicalExpr>>,
    pub aggregate: Vec<ScanAggregation>,
}

#[cfg(feature = "csv-file")]
impl Executor for CsvExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let (cache_key, cached) = cache_hit(&self.path, &self.predicate, state);
        if let Some(df) = cached {
            return Ok(df);
        }

        // cache miss

        let mut with_columns = mem::take(&mut self.options.with_columns);
        let mut projected_len = 0;
        with_columns.as_ref().map(|columns| {
            projected_len = columns.len();
            columns
        });

        if projected_len == 0 {
            with_columns = None;
        }
        let n_rows = set_n_rows(self.options.n_rows);
        let predicate = self
            .predicate
            .clone()
            .map(|expr| Arc::new(PhysicalIoHelper { expr }) as Arc<dyn PhysicalIoExpr>);

        let aggregate = if self.aggregate.is_empty() {
            None
        } else {
            Some(self.aggregate.as_slice())
        };

        let df = CsvReader::from_path(&self.path)
            .unwrap()
            .has_header(self.options.has_header)
            .with_schema(&self.schema)
            .with_delimiter(self.options.delimiter)
            .with_ignore_parser_errors(self.options.ignore_errors)
            .with_skip_rows(self.options.skip_rows)
            .with_n_rows(n_rows)
            .with_columns(with_columns)
            .low_memory(self.options.low_memory)
            .with_null_values(self.options.null_values.clone())
            .with_predicate(predicate)
            .with_aggregate(aggregate)
            .with_encoding(CsvEncoding::LossyUtf8)
            .with_comment_char(self.options.comment_char)
            .with_quote_char(self.options.quote_char)
            .finish()?;

        if self.options.cache {
            state.store_cache(cache_key, df.clone());
        }
        if state.verbose {
            println!("csv {:?} read", self.path);
        }

        Ok(df)
    }
}

/// Producer of an in memory DataFrame
pub struct DataFrameExec {
    df: Arc<DataFrame>,
    projection: Option<Vec<Arc<dyn PhysicalExpr>>>,
    selection: Option<Arc<dyn PhysicalExpr>>,
}

impl DataFrameExec {
    pub(crate) fn new(
        df: Arc<DataFrame>,
        projection: Option<Vec<Arc<dyn PhysicalExpr>>>,
        selection: Option<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        DataFrameExec {
            df,
            projection,
            selection,
        }
    }
}

impl Executor for DataFrameExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let df = mem::take(&mut self.df);
        let mut df = Arc::try_unwrap(df).unwrap_or_else(|df| (*df).clone());

        // projection should be before selection as those are free
        // TODO: this is only the case if we don't create new columns
        if let Some(projection) = &self.projection {
            df = evaluate_physical_expressions(&df, projection, state)?;
        }

        if let Some(selection) = &self.selection {
            let s = selection.evaluate(&df, state)?;
            let mask = s.bool().map_err(|_| {
                PolarsError::ComputeError("filter predicate was not of type boolean".into())
            })?;
            df = df.filter(mask)?;
        }

        if let Some(limit) = set_n_rows(None) {
            Ok(df.head(Some(limit)))
        } else {
            Ok(df)
        }
    }
}
