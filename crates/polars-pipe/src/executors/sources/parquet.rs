use std::collections::VecDeque;
use std::ops::{Deref, Range};
use std::path::PathBuf;
use std::sync::Arc;

use polars_core::config::get_file_prefetch_size;
use polars_core::error::*;
use polars_core::POOL;
use polars_io::cloud::CloudOptions;
use polars_io::parquet::{BatchedParquetReader, FileMetaData, ParquetReader};
use polars_io::pl_async::get_runtime;
use polars_io::prelude::materialize_projection;
#[cfg(feature = "async")]
use polars_io::prelude::ParquetAsyncReader;
use polars_io::{is_cloud_url, SerReader};
use polars_plan::logical_plan::FileInfo;
use polars_plan::prelude::{FileScanOptions, ParquetOptions};
use polars_utils::IdxSize;

use crate::operators::{DataChunk, PExecutionContext, Source, SourceResult};
use crate::pipeline::determine_chunk_size;

pub struct ParquetSource {
    batched_readers: VecDeque<BatchedParquetReader>,
    n_threads: usize,
    processed_paths: usize,
    chunk_index: IdxSize,
    iter: Range<usize>,
    paths: Arc<[PathBuf]>,
    options: ParquetOptions,
    file_options: FileScanOptions,
    #[allow(dead_code)]
    cloud_options: Option<CloudOptions>,
    metadata: Option<Arc<FileMetaData>>,
    file_info: FileInfo,
    verbose: bool,
    prefetch_size: usize,
}

impl ParquetSource {
    fn init_next_reader(&mut self) -> PolarsResult<()> {
        self.metadata = None;
        self.init_reader()
    }

    fn init_reader(&mut self) -> PolarsResult<()> {
        let Some(index) = self.iter.next() else {
            return Ok(());
        };
        let path = &self.paths[index];
        let options = self.options;
        let file_options = self.file_options.clone();
        let schema = self.file_info.schema.clone();

        self.file_info.update_hive_partitions(path);
        let hive_partitions = self
            .file_info
            .hive_parts
            .as_ref()
            .map(|hive| hive.materialize_partition_columns());

        let projection = materialize_projection(
            file_options
                .with_columns
                .as_deref()
                .map(|cols| cols.deref()),
            &schema,
            hive_partitions.as_deref(),
            false,
        );

        let n_cols = projection.as_ref().map(|v| v.len()).unwrap_or(schema.len());
        let chunk_size = determine_chunk_size(n_cols, self.n_threads)?;

        if self.verbose {
            eprintln!("STREAMING CHUNK SIZE: {chunk_size} rows")
        }

        let reader_schema = if self.processed_paths == 0 {
            self.file_info.reader_schema.clone()
        } else {
            None
        };

        let batched_reader = if is_cloud_url(path) {
            #[cfg(not(feature = "async"))]
            {
                panic!(
                    "Feature 'async' (or more likely one of the cloud provider features) is required to access parquet files on cloud storage."
                )
            }
            #[cfg(feature = "async")]
            {
                let uri = path.to_string_lossy();
                polars_io::pl_async::get_runtime().block_on(async {
                    ParquetAsyncReader::from_uri(
                        &uri,
                        self.cloud_options.as_ref(),
                        reader_schema,
                        self.metadata.clone(),
                    )
                    .await?
                    .with_n_rows(file_options.n_rows)
                    .with_row_count(file_options.row_count)
                    .with_projection(projection)
                    .use_statistics(options.use_statistics)
                    .with_hive_partition_columns(hive_partitions)
                    .batched(chunk_size)
                    .await
                })?
            }
        } else {
            let file = std::fs::File::open(path).unwrap();

            ParquetReader::new(file)
                .with_schema(reader_schema)
                .with_n_rows(file_options.n_rows)
                .with_row_count(file_options.row_count)
                .with_projection(projection)
                .use_statistics(options.use_statistics)
                .with_hive_partition_columns(hive_partitions)
                .batched(chunk_size)?
        };
        if self.processed_paths >= 1 {
            polars_ensure!(batched_reader.schema().as_ref() == self.file_info.reader_schema.as_ref().unwrap().as_ref(), ComputeError: "schema of all files in a single scan_parquet must be equal");
        }
        self.batched_readers.push_back(batched_reader);
        self.processed_paths += 1;
        Ok(())
    }

    #[allow(unused_variables)]
    pub(crate) fn new(
        paths: Arc<[PathBuf]>,
        options: ParquetOptions,
        cloud_options: Option<CloudOptions>,
        metadata: Option<Arc<FileMetaData>>,
        file_options: FileScanOptions,
        file_info: FileInfo,
        verbose: bool,
    ) -> PolarsResult<Self> {
        let n_threads = POOL.current_num_threads();

        let iter = 0..paths.len();

        let prefetch_size = get_file_prefetch_size();

        let mut source = ParquetSource {
            batched_readers: VecDeque::new(),
            n_threads,
            chunk_index: 0,
            processed_paths: 0,
            options,
            file_options,
            iter,
            paths,
            cloud_options,
            metadata,
            file_info,
            verbose,
            prefetch_size,
        };
        // Already start downloading when we deal with cloud urls.
        if !source.paths.first().unwrap().is_file() {
            source.init_reader()?;
        }
        Ok(source)
    }
}

impl Source for ParquetSource {
    fn get_batches(&mut self, _context: &PExecutionContext) -> PolarsResult<SourceResult> {
        // We already start downloading the next file, we can only do that if we don't have a limit.
        // In the case of a limit we first must update the row count with the batch results.
        if self.batched_readers.len() < self.prefetch_size && self.file_options.n_rows.is_none()
            || self.batched_readers.is_empty()
        {
            self.init_next_reader()?
        }

        let Some(mut reader) = self.batched_readers.pop_front() else {
            // If there was no new reader, we depleted all of them and are finished.
            return Ok(SourceResult::Finished);
        };

        let batches = get_runtime().block_on(reader.next_batches(self.n_threads))?;
        Ok(match batches {
            None => {
                if reader.limit_reached() {
                    return Ok(SourceResult::Finished);
                }

                // reset the reader
                self.init_next_reader()?;
                return self.get_batches(_context);
            },
            Some(batches) => {
                let result = SourceResult::GotMoreData(
                    batches
                        .into_iter()
                        .map(|data| {
                            // Keep the row limit updated so the next reader will have a correct limit.
                            if let Some(n_rows) = &mut self.file_options.n_rows {
                                *n_rows = n_rows.saturating_sub(data.height())
                            }

                            let chunk_index = self.chunk_index;
                            self.chunk_index += 1;
                            DataChunk { chunk_index, data }
                        })
                        .collect(),
                );
                // We are not yet done with this reader.
                // Ensure it is used in next iteration.
                self.batched_readers.push_front(reader);

                result
            },
        })
    }
    fn fmt(&self) -> &str {
        "parquet"
    }
}
