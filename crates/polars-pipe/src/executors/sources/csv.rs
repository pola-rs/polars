use std::fs::File;
use std::path::PathBuf;

use polars_core::{config, POOL};
use polars_io::csv::read::{BatchedCsvReader, CsvReadOptions, CsvReader};
use polars_io::path_utils::is_cloud_url;
use polars_plan::global::_set_n_rows_for_scan;
use polars_plan::prelude::FileScanOptions;
use polars_utils::itertools::Itertools;

use super::*;
use crate::pipeline::determine_chunk_size;

pub(crate) struct CsvSource {
    #[allow(dead_code)]
    // this exist because we need to keep ownership
    schema: SchemaRef,
    // Safety: `reader` outlives `batched_reader`
    // (so we have to order the `batched_reader` first in the struct fields)
    batched_reader: Option<BatchedCsvReader<'static>>,
    reader: Option<CsvReader<File>>,
    n_threads: usize,
    paths: Arc<Vec<PathBuf>>,
    options: Option<CsvReadOptions>,
    file_options: FileScanOptions,
    verbose: bool,
    // state for multi-file reads
    current_path_idx: usize,
    n_rows_read: usize,
    first_schema: Schema,
    include_file_path: Option<StringChunked>,
}

impl CsvSource {
    // Delay initializing the reader
    // otherwise all files would be opened during construction of the pipeline
    // leading to Too many Open files error
    fn init_next_reader(&mut self) -> PolarsResult<()> {
        let file_options = self.file_options.clone();

        let n_rows = file_options.slice.map(|x| {
            assert_eq!(x.0, 0);
            x.1
        });

        if self.current_path_idx == self.paths.len()
            || (n_rows.is_some() && n_rows.unwrap() <= self.n_rows_read)
        {
            return Ok(());
        }
        let path = &self.paths[self.current_path_idx];

        let force_async = config::force_async();
        let run_async = force_async || is_cloud_url(path);

        if self.current_path_idx == 0 && force_async && self.verbose {
            eprintln!("ASYNC READING FORCED");
        }

        self.current_path_idx += 1;

        let options = self.options.clone().unwrap();
        let mut with_columns = file_options.with_columns;
        let mut projected_len = 0;
        with_columns
            .as_ref()
            .inspect(|columns| projected_len = columns.len());

        if projected_len == 0 {
            with_columns = None;
        }

        let n_cols = if projected_len > 0 {
            projected_len
        } else {
            self.schema.len()
        };
        let n_rows = _set_n_rows_for_scan(
            file_options
                .slice
                .map(|x| {
                    assert_eq!(x.0, 0);
                    x.1
                })
                .map(|n| n.saturating_sub(self.n_rows_read)),
        );
        let row_index = file_options.row_index.map(|mut ri| {
            ri.offset += self.n_rows_read as IdxSize;
            ri
        });
        // inversely scale the chunk size by the number of threads so that we reduce memory pressure
        // in streaming
        let chunk_size = determine_chunk_size(n_cols, POOL.current_num_threads())?;

        if self.verbose {
            eprintln!("STREAMING CHUNK SIZE: {chunk_size} rows")
        }

        let options = options
            .with_schema(Some(self.schema.clone()))
            .with_n_rows(n_rows)
            .with_columns(with_columns)
            .with_rechunk(false)
            .with_row_index(row_index);

        let reader: CsvReader<File> = if run_async {
            #[cfg(feature = "cloud")]
            {
                options.into_reader_with_file_handle(
                    polars_io::file_cache::FILE_CACHE
                        .get_entry(path.to_str().unwrap())
                        // Safety: This was initialized by schema inference.
                        .unwrap()
                        .try_open_assume_latest()?,
                )
            }
            #[cfg(not(feature = "cloud"))]
            {
                panic!("required feature `cloud` is not enabled")
            }
        } else {
            options
                .with_path(Some(path))
                .try_into_reader_with_file_path(None)?
        };

        if let Some(col) = &file_options.include_file_paths {
            self.include_file_path = Some(StringChunked::full(col, path.to_str().unwrap(), 1));
        };

        self.reader = Some(reader);
        let reader = self.reader.as_mut().unwrap();

        // Safety: `reader` outlives `batched_reader`
        let reader: &'static mut CsvReader<File> = unsafe { std::mem::transmute(reader) };
        let batched_reader = reader.batched_borrowed()?;
        self.batched_reader = Some(batched_reader);
        Ok(())
    }

    pub(crate) fn new(
        paths: Arc<Vec<PathBuf>>,
        schema: SchemaRef,
        options: CsvReadOptions,
        file_options: FileScanOptions,
        verbose: bool,
    ) -> PolarsResult<Self> {
        Ok(CsvSource {
            schema,
            reader: None,
            batched_reader: None,
            n_threads: POOL.current_num_threads(),
            paths,
            options: Some(options),
            file_options,
            verbose,
            current_path_idx: 0,
            n_rows_read: 0,
            first_schema: Default::default(),
            include_file_path: None,
        })
    }
}

impl Source for CsvSource {
    fn get_batches(&mut self, _context: &PExecutionContext) -> PolarsResult<SourceResult> {
        loop {
            let first_read_from_file = self.reader.is_none();

            if first_read_from_file {
                self.init_next_reader()?;
            }

            if self.reader.is_none() {
                // No more readers
                return Ok(SourceResult::Finished);
            }

            let Some(batches) = self
                .batched_reader
                .as_mut()
                .unwrap()
                .next_batches(self.n_threads)?
            else {
                self.reader = None;
                continue;
            };

            if first_read_from_file {
                if self.first_schema.is_empty() {
                    self.first_schema = batches[0].schema();
                }
                ensure_matching_schema(&self.first_schema, &batches[0].schema())?;
            }

            let index = get_source_index(0);
            let mut n_rows_read = 0;
            let mut max_height = 0;
            let mut out = batches
                .into_iter()
                .enumerate_u32()
                .map(|(i, data)| {
                    max_height = max_height.max(data.height());
                    n_rows_read += data.height();
                    DataChunk {
                        chunk_index: (index + i) as IdxSize,
                        data,
                    }
                })
                .collect::<Vec<_>>();

            if let Some(ca) = &mut self.include_file_path {
                if ca.len() < max_height {
                    *ca = ca.new_from_index(0, max_height);
                };

                for data_chunk in &mut out {
                    // The batched reader creates the column containing all nulls because the schema it
                    // gets passed contains the column.
                    for s in unsafe { data_chunk.data.get_columns_mut() } {
                        if s.name() == ca.name() {
                            *s = ca.slice(0, s.len()).into_series();
                            break;
                        }
                    }
                }
            }

            self.n_rows_read = self.n_rows_read.saturating_add(n_rows_read);
            get_source_index(out.len() as u32);

            return Ok(SourceResult::GotMoreData(out));
        }
    }
    fn fmt(&self) -> &str {
        "csv"
    }
}
