use std::fs::File;
use std::path::PathBuf;

use polars_core::export::arrow::Either;
use polars_core::POOL;
use polars_io::csv::read_impl::{BatchedCsvReaderMmap, BatchedCsvReaderRead};
use polars_io::csv::{CsvEncoding, CsvReader};
use polars_plan::global::_set_n_rows_for_scan;
use polars_plan::prelude::CsvParserOptions;

use super::*;
use crate::pipeline::determine_chunk_size;

pub(crate) struct CsvSource {
    #[allow(dead_code)]
    // this exist because we need to keep ownership
    schema: SchemaRef,
    reader: Option<*mut CsvReader<'static, File>>,
    batched_reader:
        Option<Either<*mut BatchedCsvReaderMmap<'static>, *mut BatchedCsvReaderRead<'static>>>,
    n_threads: usize,
    chunk_index: IdxSize,
    path: Option<PathBuf>,
    options: Option<CsvParserOptions>,
    verbose: bool,
}

impl CsvSource {
    // Delay initializing the reader
    // otherwise all files would be opened during construction of the pipeline
    // leading to Too many Open files error
    fn init_reader(&mut self) -> PolarsResult<()> {
        let options = self.options.take().unwrap();
        let path = self.path.take().unwrap();
        let mut with_columns = options.with_columns;
        let mut projected_len = 0;
        with_columns.as_ref().map(|columns| {
            projected_len = columns.len();
            columns
        });

        if projected_len == 0 {
            with_columns = None;
        }

        let n_cols = if projected_len > 0 {
            projected_len
        } else {
            self.schema.len()
        };
        let n_rows = _set_n_rows_for_scan(options.n_rows);
        // inversely scale the chunk size by the number of threads so that we reduce memory pressure
        // in streaming
        let chunk_size = determine_chunk_size(n_cols, POOL.current_num_threads())?;

        if self.verbose {
            eprintln!("STREAMING CHUNK SIZE: {chunk_size} rows")
        }

        let reader = CsvReader::from_path(&path)
            .unwrap()
            .has_header(options.has_header)
            .with_schema(self.schema.clone())
            .with_delimiter(options.delimiter)
            .with_ignore_errors(options.ignore_errors)
            .with_skip_rows(options.skip_rows)
            .with_n_rows(n_rows)
            .with_columns(with_columns.map(|mut cols| std::mem::take(Arc::make_mut(&mut cols))))
            .low_memory(options.low_memory)
            .with_null_values(options.null_values)
            .with_encoding(CsvEncoding::LossyUtf8)
            .with_comment_char(options.comment_char)
            .with_quote_char(options.quote_char)
            .with_end_of_line_char(options.eol_char)
            .with_encoding(options.encoding)
            // never rechunk in streaming
            .with_rechunk(false)
            .with_chunk_size(chunk_size)
            .with_row_count(options.row_count)
            .with_try_parse_dates(options.try_parse_dates);

        let reader = Box::new(reader);
        let reader = Box::leak(reader) as *mut CsvReader<'static, File>;

        let batched_reader = if options.low_memory {
            let batched_reader = unsafe { Box::new((*reader).batched_borrowed_read()?) };
            let batched_reader = Box::leak(batched_reader) as *mut BatchedCsvReaderRead;
            Either::Right(batched_reader)
        } else {
            let batched_reader = unsafe { Box::new((*reader).batched_borrowed_mmap()?) };
            let batched_reader = Box::leak(batched_reader) as *mut BatchedCsvReaderMmap;
            Either::Left(batched_reader)
        };
        self.reader = Some(reader);
        self.batched_reader = Some(batched_reader);
        Ok(())
    }

    pub(crate) fn new(
        path: PathBuf,
        schema: SchemaRef,
        options: CsvParserOptions,
        verbose: bool,
    ) -> PolarsResult<Self> {
        Ok(CsvSource {
            schema,
            reader: None,
            batched_reader: None,
            n_threads: POOL.current_num_threads(),
            chunk_index: 0,
            path: Some(path),
            options: Some(options),
            verbose,
        })
    }
}

impl Drop for CsvSource {
    fn drop(&mut self) {
        unsafe {
            match self.batched_reader {
                Some(Either::Left(ptr)) => {
                    let _to_drop = Box::from_raw(ptr);
                }
                Some(Either::Right(ptr)) => {
                    let _to_drop = Box::from_raw(ptr);
                }
                // nothing initialized, nothing to drop
                _ => {}
            }
            if let Some(ptr) = self.reader {
                let _to_drop = Box::from_raw(ptr);
            }
        };
    }
}

unsafe impl Send for CsvSource {}
unsafe impl Sync for CsvSource {}

impl Source for CsvSource {
    fn get_batches(&mut self, _context: &PExecutionContext) -> PolarsResult<SourceResult> {
        if self.reader.is_none() {
            self.init_reader()?
        }

        let batches = match self.batched_reader.unwrap() {
            Either::Left(batched_reader) => {
                let reader = unsafe { &mut *batched_reader };

                reader.next_batches(self.n_threads)?
            }
            Either::Right(batched_reader) => {
                let reader = unsafe { &mut *batched_reader };

                reader.next_batches(self.n_threads)?
            }
        };
        Ok(match batches {
            None => SourceResult::Finished,
            Some(batches) => SourceResult::GotMoreData(
                batches
                    .into_iter()
                    .map(|data| {
                        let out = DataChunk {
                            chunk_index: self.chunk_index,
                            data,
                        };
                        self.chunk_index += 1;
                        out
                    })
                    .collect(),
            ),
        })
    }
    fn fmt(&self) -> &str {
        "csv"
    }
}
