use std::collections::VecDeque;
use std::ops::{Deref, Range};
use std::path::PathBuf;
use std::sync::Arc;

use arrow::datatypes::ArrowSchemaRef;
use polars_core::config::{self, get_file_prefetch_size};
use polars_core::error::*;
use polars_core::prelude::Series;
use polars_core::POOL;
use polars_io::cloud::CloudOptions;
use polars_io::parquet::metadata::FileMetaDataRef;
use polars_io::parquet::read::{BatchedParquetReader, ParquetOptions, ParquetReader};
use polars_io::pl_async::get_runtime;
use polars_io::predicates::PhysicalIoExpr;
use polars_io::prelude::materialize_projection;
#[cfg(feature = "async")]
use polars_io::prelude::ParquetAsyncReader;
use polars_io::utils::{check_projected_arrow_schema, is_cloud_url};
use polars_io::SerReader;
use polars_plan::logical_plan::FileInfo;
use polars_plan::prelude::FileScanOptions;
use polars_utils::iter::EnumerateIdxTrait;
use polars_utils::IdxSize;

use crate::executors::sources::get_source_index;
use crate::operators::{DataChunk, PExecutionContext, Source, SourceResult};
use crate::pipeline::determine_chunk_size;

pub struct ParquetSource {
    batched_readers: VecDeque<BatchedParquetReader>,
    n_threads: usize,
    processed_paths: usize,
    iter: Range<usize>,
    paths: Arc<[PathBuf]>,
    options: ParquetOptions,
    file_options: FileScanOptions,
    #[allow(dead_code)]
    cloud_options: Option<CloudOptions>,
    metadata: Option<FileMetaDataRef>,
    file_info: FileInfo,
    verbose: bool,
    run_async: bool,
    prefetch_size: usize,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
}

impl ParquetSource {
    fn init_next_reader(&mut self) -> PolarsResult<()> {
        if !self.run_async {
            // Don't do this for async as that would mean we run serially.
            self.init_next_reader_sync()
        } else {
            Ok(())
        }
    }

    fn init_next_reader_sync(&mut self) -> PolarsResult<()> {
        self.metadata = None;
        self.init_reader_sync()
    }

    #[allow(clippy::type_complexity)]
    fn prepare_init_reader(
        &self,
        index: usize,
    ) -> PolarsResult<(
        &PathBuf,
        ParquetOptions,
        FileScanOptions,
        Option<Vec<usize>>,
        usize,
        Option<ArrowSchemaRef>,
        Option<Vec<Series>>,
    )> {
        let path = &self.paths[index];
        let options = self.options;
        let file_options = self.file_options.clone();
        let schema = self.file_info.schema.clone();

        let mut file_info = self.file_info.clone();
        file_info.update_hive_partitions(path)?;
        let hive_partitions = file_info
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
        Ok((
            path,
            options,
            file_options,
            projection,
            chunk_size,
            reader_schema.map(|either| either.unwrap_left()),
            hive_partitions,
        ))
    }

    fn init_reader_sync(&mut self) -> PolarsResult<()> {
        let Some(index) = self.iter.next() else {
            return Ok(());
        };
        let predicate = self.predicate.clone();
        let (path, options, file_options, projection, chunk_size, reader_schema, hive_partitions) =
            self.prepare_init_reader(index)?;

        let batched_reader = {
            let file = std::fs::File::open(path).unwrap();
            ParquetReader::new(file)
                .with_schema(reader_schema)
                .with_n_rows(file_options.n_rows)
                .with_row_index(file_options.row_index)
                .with_predicate(predicate.clone())
                .with_projection(projection)
                .use_statistics(options.use_statistics)
                .with_hive_partition_columns(hive_partitions)
                .batched(chunk_size)?
        };
        self.finish_init_reader(batched_reader)?;
        Ok(())
    }

    fn finish_init_reader(&mut self, batched_reader: BatchedParquetReader) -> PolarsResult<()> {
        if self.processed_paths >= 1 {
            let with_columns = self
                .file_options
                .with_columns
                .as_ref()
                .map(|v| v.as_slice());
            check_projected_arrow_schema(
                batched_reader.schema().as_ref(),
                self.file_info
                    .reader_schema
                    .as_ref()
                    .unwrap()
                    .as_ref()
                    .unwrap_left(),
                with_columns,
                "schema of all files in a single scan_parquet must be equal",
            )?;
        }
        self.batched_readers.push_back(batched_reader);
        self.processed_paths += 1;
        Ok(())
    }

    #[cfg(feature = "async")]
    async fn init_reader_async(&self, index: usize) -> PolarsResult<BatchedParquetReader> {
        let metadata = self.metadata.clone();
        let predicate = self.predicate.clone();
        let cloud_options = self.cloud_options.clone();
        let (path, options, file_options, projection, chunk_size, reader_schema, hive_partitions) =
            self.prepare_init_reader(index)?;

        let batched_reader = {
            let uri = path.to_string_lossy();
            ParquetAsyncReader::from_uri(&uri, cloud_options.as_ref(), reader_schema, metadata)
                .await?
                .with_n_rows(file_options.n_rows)
                .with_row_index(file_options.row_index)
                .with_projection(projection)
                .with_predicate(predicate.clone())
                .use_statistics(options.use_statistics)
                .with_hive_partition_columns(hive_partitions)
                .batched(chunk_size)
                .await?
        };
        Ok(batched_reader)
    }

    #[allow(unused_variables)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        paths: Arc<[PathBuf]>,
        options: ParquetOptions,
        cloud_options: Option<CloudOptions>,
        metadata: Option<FileMetaDataRef>,
        file_options: FileScanOptions,
        file_info: FileInfo,
        verbose: bool,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
    ) -> PolarsResult<Self> {
        let n_threads = POOL.current_num_threads();

        let iter = 0..paths.len();

        let prefetch_size = get_file_prefetch_size();
        if verbose {
            eprintln!("POLARS PREFETCH_SIZE: {}", prefetch_size)
        }
        let run_async = paths.first().map(is_cloud_url).unwrap_or(false) || config::force_async();

        let mut source = ParquetSource {
            batched_readers: VecDeque::new(),
            n_threads,
            processed_paths: 0,
            options,
            file_options,
            iter,
            paths,
            cloud_options,
            metadata,
            file_info,
            verbose,
            run_async,
            prefetch_size,
            predicate,
        };
        // Already start downloading when we deal with cloud urls.
        if run_async {
            source.init_next_reader()?;
            source.metadata = None;
        }
        Ok(source)
    }

    fn prefetch_files(&mut self) -> PolarsResult<()> {
        // We already start downloading the next file, we can only do that if we don't have a limit.
        // In the case of a limit we first must update the row count with the batch results.
        //
        // It is important we do this for a reasonable batch size, that's why we start this when we
        // have just 2 readers left.
        if self.batched_readers.len() <= 2 && self.file_options.n_rows.is_none()
            || self.batched_readers.is_empty()
        {
            let range = 0..self.prefetch_size - self.batched_readers.len();

            if self.run_async {
                #[cfg(not(feature = "async"))]
                panic!("activate 'async' feature");

                #[cfg(feature = "async")]
                {
                    let range = range
                        .zip(&mut self.iter)
                        .map(|(_, index)| index)
                        .collect::<Vec<_>>();
                    let init_iter = range.into_iter().map(|index| self.init_reader_async(index));

                    let batched_readers = polars_io::pl_async::get_runtime()
                        .block_on_potential_spawn(async {
                            futures::future::try_join_all(init_iter).await
                        })?;

                    for r in batched_readers {
                        self.finish_init_reader(r)?;
                    }
                }
            } else {
                for _ in 0..self.prefetch_size - self.batched_readers.len() {
                    self.init_next_reader()?
                }
            }
        }
        Ok(())
    }
}

impl Source for ParquetSource {
    fn get_batches(&mut self, _context: &PExecutionContext) -> PolarsResult<SourceResult> {
        self.prefetch_files()?;

        let Some(mut reader) = self.batched_readers.pop_front() else {
            // If there was no new reader, we depleted all of them and are finished.
            return Ok(SourceResult::Finished);
        };

        let batches =
            get_runtime().block_on_potential_spawn(reader.next_batches(self.n_threads))?;

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
                let idx_offset = get_source_index(0);
                let out = batches
                    .into_iter()
                    .enumerate_u32()
                    .map(|(i, data)| {
                        // Keep the row limit updated so the next reader will have a correct limit.
                        if let Some(n_rows) = &mut self.file_options.n_rows {
                            *n_rows = n_rows.saturating_sub(data.height())
                        }

                        DataChunk {
                            chunk_index: (idx_offset + i) as IdxSize,
                            data,
                        }
                    })
                    .collect::<Vec<_>>();
                get_source_index(out.len() as u32);

                let result = SourceResult::GotMoreData(out);
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
