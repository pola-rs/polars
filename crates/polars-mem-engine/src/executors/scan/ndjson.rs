use polars_core::config;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_io::prelude::{JsonLineReader, SerReader};
use polars_io::utils::compression::maybe_decompress_bytes;

use super::*;

pub struct JsonExec {
    sources: ScanSources,
    options: NDJsonReadOptions,
    file_options: FileScanOptions,
    file_info: FileInfo,
    predicate: Option<Arc<dyn PhysicalExpr>>,
}

impl JsonExec {
    pub fn new(
        sources: ScanSources,
        options: NDJsonReadOptions,
        file_options: FileScanOptions,
        file_info: FileInfo,
        predicate: Option<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        Self {
            sources,
            options,
            file_options,
            file_info,
            predicate,
        }
    }

    fn read_impl(&mut self) -> PolarsResult<DataFrame> {
        let schema = self
            .file_info
            .reader_schema
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap_right();

        let verbose = config::verbose();
        let force_async = config::force_async();
        let run_async = (self.sources.is_paths() && force_async) || self.sources.is_cloud_url();

        if self.sources.is_paths() && force_async && verbose {
            eprintln!("ASYNC READING FORCED");
        }

        let mut n_rows = self.file_options.slice.map(|x| {
            assert_eq!(x.0, 0);
            x.1
        });

        // Avoid panicking
        if n_rows == Some(0) {
            let mut df = DataFrame::empty_with_schema(schema);
            if let Some(col) = &self.file_options.include_file_paths {
                unsafe {
                    df.with_column_unchecked(Column::new_empty(col.clone(), &DataType::String))
                };
            }
            if let Some(row_index) = &self.file_options.row_index {
                df.with_row_index_mut(row_index.name.clone(), Some(row_index.offset));
            }
            return Ok(df);
        }

        let dfs = self
            .sources
            .iter()
            .map_while(|source| {
                if n_rows == Some(0) {
                    return None;
                }

                let row_index = self.file_options.row_index.as_mut();

                let memslice = match source.to_memslice_async_assume_latest(run_async) {
                    Ok(memslice) => memslice,
                    Err(err) => return Some(Err(err)),
                };

                let owned = &mut vec![];
                let curs = std::io::Cursor::new(match maybe_decompress_bytes(&memslice, owned) {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                });
                let reader = JsonLineReader::new(curs);

                let df = reader
                    .with_schema(schema.clone())
                    .with_rechunk(self.file_options.rechunk)
                    .with_chunk_size(Some(self.options.chunk_size))
                    .with_row_index(row_index)
                    .with_predicate(self.predicate.clone().map(phys_expr_to_io_expr))
                    .with_projection(self.file_options.with_columns.clone())
                    .low_memory(self.options.low_memory)
                    .with_n_rows(n_rows)
                    .with_ignore_errors(self.options.ignore_errors)
                    .finish();

                let mut df = match df {
                    Ok(df) => df,
                    Err(e) => return Some(Err(e)),
                };

                if let Some(ref mut n_rows) = n_rows {
                    *n_rows -= df.height();
                }

                if let Some(col) = &self.file_options.include_file_paths {
                    let name = source.to_include_path_name();
                    unsafe {
                        df.with_column_unchecked(Column::new_scalar(
                            col.clone(),
                            Scalar::new(DataType::String, AnyValue::StringOwned(name.into())),
                            df.height(),
                        ))
                    };
                }

                Some(Ok(df))
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        accumulate_dataframes_vertical(dfs)
    }
}

impl ScanExec for JsonExec {
    fn read(
        &mut self,
        with_columns: Option<Arc<[PlSmallStr]>>,
        slice: Option<(usize, usize)>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        row_index: Option<polars_io::RowIndex>,
    ) -> PolarsResult<DataFrame> {
        self.file_options.with_columns = with_columns;
        self.file_options.slice = slice.map(|(s, l)| (s as i64, l));
        self.predicate = predicate;
        self.file_options.row_index = row_index;

        if self.file_info.reader_schema.is_none() {
            self.schema()?;
        }
        self.read_impl()
    }

    fn schema(&mut self) -> PolarsResult<&SchemaRef> {
        if self.file_info.reader_schema.is_some() {
            return Ok(&self.file_info.schema);
        }

        let memslice = self
            .sources
            .at(0)
            .to_memslice_async_assume_latest(self.sources.is_cloud_url())?;

        let owned = &mut vec![];
        let bytes = maybe_decompress_bytes(&memslice[..], owned)?;

        let schema = polars_io::ndjson::infer_schema(
            &mut std::io::Cursor::new(bytes),
            self.options.infer_schema_length,
        )?;

        let schema = Arc::new(schema);
        self.file_info.schema = schema.clone();
        self.file_info.reader_schema = Some(arrow::Either::Right(schema.clone()));

        Ok(&self.file_info.schema)
    }

    fn num_unfiltered_rows(&mut self) -> PolarsResult<IdxSize> {
        let (lb, ub) = self.file_info.row_estimation;
        if lb.is_some_and(|lb| lb == ub) {
            return Ok(ub as IdxSize);
        }

        let memslice = self
            .sources
            .at(0)
            .to_memslice_async_assume_latest(self.sources.is_cloud_url())?;

        let owned = &mut vec![];
        let bytes = maybe_decompress_bytes(&memslice[..], owned)?;

        let reader = polars_io::ndjson::core::JsonLineReader::new(std::io::Cursor::new(bytes));
        let num_unfiltered_rows = reader.count()?;

        self.file_info.row_estimation = (Some(num_unfiltered_rows), num_unfiltered_rows);

        Ok(num_unfiltered_rows as IdxSize)
    }
}

impl Executor for JsonExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let profile_name = if state.has_node_timer() {
            let ids = vec![self.sources.id()];
            let name = comma_delimited("ndjson".to_string(), &ids);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(|| self.read_impl(), profile_name)
    }
}
