use super::*;
use crate::logical_plan::CsvParserOptions;
use crate::utils::try_path_to_str;
use polars_io::mmap::MmapBytesReader;
use polars_io::prelude::*;
use polars_io::{csv::CsvEncoding, ScanAggregation};
use std::mem;

trait FinishScanOps {
    /// Read the file and create the DataFrame. Used from lazy execution
    fn finish_with_scan_ops(
        self,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        aggregate: Option<&[ScanAggregation]>,
    ) -> Result<DataFrame>;
}

impl<'a, R: 'static + Seek + Sync + Send + MmapBytesReader> FinishScanOps for CsvReader<'a, R> {
    fn finish_with_scan_ops(
        self,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        aggregate: Option<&[ScanAggregation]>,
    ) -> Result<DataFrame> {
        let predicate =
            predicate.map(|expr| Arc::new(PhysicalIoHelper { expr }) as Arc<dyn PhysicalIoExpr>);

        let rechunk = self.rechunk;
        let mut csv_reader = self.build_inner_reader()?;
        let df = csv_reader.as_df(predicate, aggregate)?;
        match rechunk {
            true => Ok(df.agg_chunks()),
            false => Ok(df),
        }
    }
}

#[cfg(feature = "parquet")]
pub struct ParquetExec {
    path: PathBuf,
    schema: SchemaRef,
    with_columns: Option<Vec<String>>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    aggregate: Vec<ScanAggregation>,
    stop_after_n_rows: Option<usize>,
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
        stop_after_n_rows: Option<usize>,
        cache: bool,
    ) -> Self {
        ParquetExec {
            path,
            schema,
            with_columns,
            predicate,
            aggregate,
            stop_after_n_rows,
            cache,
        }
    }
}

#[cfg(feature = "parquet")]
impl Executor for ParquetExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let path_str = try_path_to_str(&self.path)?;
        let cache_key = match &self.predicate {
            Some(predicate) => format!("{}{:?}", path_str, predicate.as_expression()),
            None => path_str.to_string(),
        };
        if let Some(df) = state.cache_hit(&cache_key) {
            return Ok(df);
        }
        // cache miss
        let file = std::fs::File::open(&self.path).unwrap();

        let with_columns = mem::take(&mut self.with_columns);
        let schema = mem::take(&mut self.schema);

        let projection: Option<Vec<_>> = with_columns.map(|with_columns| {
            with_columns
                .iter()
                .map(|name| schema.column_with_name(name).unwrap().0)
                .collect()
        });

        let stop_after_n_rows = set_n_rows(self.stop_after_n_rows);
        let aggregate = if self.aggregate.is_empty() {
            None
        } else {
            Some(self.aggregate.as_slice())
        };
        let predicate = self
            .predicate
            .clone()
            .map(|expr| Arc::new(PhysicalIoHelper { expr }) as Arc<dyn PhysicalIoExpr>);

        let df = ParquetReader::new(file)
            .with_stop_after_n_rows(stop_after_n_rows)
            .finish_with_scan_ops(
                predicate,
                aggregate,
                projection.as_ref().map(|v| v.as_ref()),
            )?;

        if self.cache {
            state.store_cache(cache_key, df.clone())
        }
        if std::env::var(POLARS_VERBOSE).is_ok() {
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
        let path_str = try_path_to_str(&self.path)?;
        let state_key = match &self.predicate {
            Some(predicate) => format!("{}{:?}", path_str, predicate.as_expression()),
            None => path_str.to_string(),
        };
        if self.options.cache {
            if let Some(df) = state.cache_hit(&state_key) {
                return Ok(df);
            }
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
        let stop_after_n_rows = set_n_rows(self.options.stop_after_n_rows);

        let reader = CsvReader::from_path(&self.path)
            .unwrap()
            .has_header(self.options.has_header)
            .with_schema(self.schema.clone())
            .with_delimiter(self.options.delimiter)
            .with_ignore_parser_errors(self.options.ignore_errors)
            .with_skip_rows(self.options.skip_rows)
            .with_stop_after_n_rows(stop_after_n_rows)
            .with_columns(with_columns)
            .low_memory(self.options.low_memory)
            .with_null_values(self.options.null_values.clone())
            .with_encoding(CsvEncoding::LossyUtf8);

        let aggregate = if self.aggregate.is_empty() {
            None
        } else {
            Some(self.aggregate.as_slice())
        };

        let df = reader.finish_with_scan_ops(self.predicate.clone(), aggregate)?;

        if self.options.cache {
            state.store_cache(state_key, df.clone());
        }
        if std::env::var(POLARS_VERBOSE).is_ok() {
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
        if let Some(projection) = &self.projection {
            df = evaluate_physical_expressions(&df, projection, state)?;
        }

        if let Some(selection) = &self.selection {
            let s = selection.evaluate(&df, state)?;
            let mask = s.bool().map_err(|_| {
                PolarsError::Other("filter predicate was not of type boolean".into())
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
