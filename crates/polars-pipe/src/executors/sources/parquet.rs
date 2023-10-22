use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::utils::arrow::io::parquet::read::FileMetaData;
use polars_core::POOL;
use polars_io::cloud::CloudOptions;
use polars_io::parquet::{BatchedParquetReader, ParquetReader};
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
    batched_reader: Option<BatchedParquetReader>,
    n_threads: usize,
    chunk_index: IdxSize,
    paths: std::slice::Iter<'static, PathBuf>,
    _paths_lifetime: Arc<[PathBuf]>,
    options: ParquetOptions,
    file_options: FileScanOptions,
    #[allow(dead_code)]
    cloud_options: Option<CloudOptions>,
    metadata: Option<Arc<FileMetaData>>,
    file_info: FileInfo,
    verbose: bool,
}

impl ParquetSource {
    // Delay initializing the reader
    // otherwise all files would be opened during construction of the pipeline
    // leading to Too many Open files error
    fn init_reader(&mut self) -> PolarsResult<()> {
        let Some(path) = self.paths.next() else {
            return Ok(());
        };
        let options = self.options;
        let file_options = self.file_options.clone();
        let schema = self.file_info.schema.clone();

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
                        Some(self.file_info.reader_schema.clone()),
                        self.metadata.clone(),
                    )
                    .await?
                    .with_n_rows(file_options.n_rows)
                    .with_row_count(file_options.row_count)
                    .with_projection(projection)
                    .use_statistics(options.use_statistics)
                    .with_hive_partition_columns(
                        self.file_info
                            .hive_parts
                            .as_ref()
                            .map(|hive| hive.materialize_partition_columns()),
                    )
                    .batched(chunk_size)
                    .await
                })?
            }
        } else {
            let file = std::fs::File::open(path).unwrap();

            ParquetReader::new(file)
                .with_schema(Some(self.file_info.reader_schema.clone()))
                .with_n_rows(file_options.n_rows)
                .with_row_count(file_options.row_count)
                .with_projection(projection)
                .use_statistics(options.use_statistics)
                .with_hive_partition_columns(
                    self.file_info
                        .hive_parts
                        .as_ref()
                        .map(|hive| hive.materialize_partition_columns()),
                )
                .batched(chunk_size)?
        };
        self.batched_reader = Some(batched_reader);
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

        // extend lifetime as it will be bound to parquet source
        let iter = unsafe {
            std::mem::transmute::<std::slice::Iter<'_, PathBuf>, std::slice::Iter<'static, PathBuf>>(
                paths.iter(),
            )
        };

        Ok(ParquetSource {
            batched_reader: None,
            n_threads,
            chunk_index: 0,
            options,
            file_options,
            paths: iter,
            _paths_lifetime: paths,
            cloud_options,
            metadata,
            file_info,
            verbose,
        })
    }
}

impl Source for ParquetSource {
    fn get_batches(&mut self, _context: &PExecutionContext) -> PolarsResult<SourceResult> {
        if self.batched_reader.is_none() {
            self.init_reader()?;

            // If there was no new reader, we depleted all of them and are finished.
            if self.batched_reader.is_none() {
                return Ok(SourceResult::Finished);
            }
        }
        let batches = get_runtime().block_on(
            self.batched_reader
                .as_mut()
                .unwrap()
                .next_batches(self.n_threads),
        )?;
        Ok(match batches {
            None => {
                // reset the reader
                self.batched_reader = None;
                return self.get_batches(_context);
            },
            Some(batches) => SourceResult::GotMoreData(
                batches
                    .into_iter()
                    .map(|data| {
                        let chunk_index = self.chunk_index;
                        self.chunk_index += 1;
                        DataChunk { chunk_index, data }
                    })
                    .collect(),
            ),
        })
    }
    fn fmt(&self) -> &str {
        "parquet"
    }
}
