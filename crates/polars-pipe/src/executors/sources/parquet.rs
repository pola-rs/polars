use std::collections::VecDeque;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use polars_core::config::{self, get_file_prefetch_size};
use polars_core::error::*;
use polars_core::prelude::Series;
use polars_core::POOL;
use polars_io::cloud::CloudOptions;
use polars_io::parquet::metadata::FileMetaDataRef;
use polars_io::parquet::read::{BatchedParquetReader, ParquetOptions, ParquetReader};
use polars_io::path_utils::is_cloud_url;
use polars_io::pl_async::get_runtime;
use polars_io::predicates::PhysicalIoExpr;
use polars_io::prelude::materialize_projection;
#[cfg(feature = "async")]
use polars_io::prelude::ParquetAsyncReader;
use polars_io::utils::slice::split_slice_at_file;
use polars_io::SerReader;
use polars_plan::plans::FileInfo;
use polars_plan::prelude::hive::HivePartitions;
use polars_plan::prelude::FileScanOptions;
use polars_utils::itertools::Itertools;
use polars_utils::IdxSize;

use crate::executors::sources::get_source_index;
use crate::operators::{DataChunk, PExecutionContext, Source, SourceResult};
use crate::pipeline::determine_chunk_size;

pub struct ParquetSource {
    batched_readers: VecDeque<BatchedParquetReader>,
    n_threads: usize,
    processed_paths: usize,
    processed_rows: AtomicUsize,
    iter: Range<usize>,
    paths: Arc<Vec<PathBuf>>,
    total_files_read: usize,
    options: ParquetOptions,
    file_options: FileScanOptions,
    #[allow(dead_code)]
    cloud_options: Option<CloudOptions>,
    metadata: Option<FileMetaDataRef>,
    file_info: FileInfo,
    hive_parts: Option<Arc<Vec<HivePartitions>>>,
    verbose: bool,
    run_async: bool,
    prefetch_size: usize,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    rows_left_to_read: usize,
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
        Option<Vec<Series>>,
    )> {
        let path = &self.paths[index];
        let options = self.options;
        let file_options = self.file_options.clone();
        let schema = self.file_info.schema.clone();

        let hive_partitions = self
            .hive_parts
            .as_ref()
            .map(|x| x[index].materialize_partition_columns());

        let projection = materialize_projection(
            file_options.with_columns.as_deref(),
            &schema,
            hive_partitions.as_deref(),
            false,
        );

        let n_cols = projection.as_ref().map(|v| v.len()).unwrap_or(schema.len());
        let chunk_size = determine_chunk_size(n_cols, self.n_threads)?;

        if self.verbose {
            eprintln!("STREAMING CHUNK SIZE: {chunk_size} rows")
        }

        Ok((
            path,
            options,
            file_options,
            projection,
            chunk_size,
            hive_partitions,
        ))
    }

    fn init_reader_sync(&mut self) -> PolarsResult<()> {
        use std::sync::atomic::Ordering;

        let Some(index) = self.iter.next() else {
            return Ok(());
        };
        if let Some(slice) = self.file_options.slice {
            if self.processed_rows.load(Ordering::Relaxed) >= slice.0 as usize + slice.1 {
                return Ok(());
            }
        }

        let predicate = self.predicate.clone();
        let (path, options, file_options, projection, chunk_size, hive_partitions) =
            self.prepare_init_reader(index)?;

        let batched_reader = {
            let file = std::fs::File::open(path).unwrap();
            let mut reader = ParquetReader::new(file)
                .with_projection(projection)
                .check_schema(
                    self.file_info
                        .reader_schema
                        .as_ref()
                        .unwrap()
                        .as_ref()
                        .unwrap_left(),
                )?
                .with_row_index(file_options.row_index)
                .with_predicate(predicate.clone())
                .use_statistics(options.use_statistics)
                .with_hive_partition_columns(hive_partitions)
                .with_include_file_path(
                    self.file_options
                        .include_file_paths
                        .as_ref()
                        .map(|x| (x.clone(), Arc::from(path.to_str().unwrap()))),
                );

            let n_rows_this_file = reader.num_rows().unwrap();
            let current_row_offset = self
                .processed_rows
                .fetch_add(n_rows_this_file, Ordering::Relaxed);

            let slice = file_options.slice.map(|slice| {
                assert!(slice.0 >= 0);
                let slice_start = slice.0 as usize;
                let slice_end = slice_start + slice.1;
                split_slice_at_file(
                    &mut current_row_offset.clone(),
                    n_rows_this_file,
                    slice_start,
                    slice_end,
                )
            });

            reader = reader.with_slice(slice);
            reader.batched(chunk_size)?
        };
        self.finish_init_reader(batched_reader)?;
        Ok(())
    }

    fn finish_init_reader(&mut self, batched_reader: BatchedParquetReader) -> PolarsResult<()> {
        self.batched_readers.push_back(batched_reader);
        self.processed_paths += 1;
        Ok(())
    }

    /// This function must NOT be run concurrently if there is a slice (or any operation that
    /// requires `self.processed_rows` to be incremented in the correct order), as it does not
    /// coordinate to increment the row offset in a properly ordered manner.
    #[cfg(feature = "async")]
    async fn init_reader_async(
        &self,
        index: usize,
        n_rows: usize,
    ) -> PolarsResult<(ParquetAsyncReader, usize)> {
        let metadata: Option<Arc<polars_io::prelude::FileMetaData>> = self.metadata.clone();
        let predicate = self.predicate.clone();
        let cloud_options = self.cloud_options.clone();
        let (path, options, file_options, projection, chunk_size, hive_partitions) =
            self.prepare_init_reader(index)?;

        let reader = {
            let uri = path.to_string_lossy();
            ParquetAsyncReader::from_uri(&uri, cloud_options.as_ref(), metadata)
                .await?
                .read_parallel(options.parallel)
                .with_row_index(file_options.row_index)
                .with_projection(projection)
                .with_predicate(predicate.clone())
                .with_hive_partition_columns(hive_partitions)
                .use_statistics(options.use_statistics)
                .with_slice(Some((0, n_rows)))
                .with_include_file_path(
                    self.file_options
                        .include_file_paths
                        .as_ref()
                        .map(|x| (x.clone(), Arc::from(path.to_str().unwrap()))),
                )
        };
        Ok((reader, chunk_size))
    }

    #[cfg(feature = "async")]
    async fn init_batch_reader(
        &self,
        reader: ParquetAsyncReader,
        chunk_size: usize,
    ) -> PolarsResult<BatchedParquetReader> {
        let batched_reader = reader
            .check_schema(
                self.file_info
                    .reader_schema
                    .as_ref()
                    .unwrap()
                    .as_ref()
                    .unwrap_left(),
            )
            .await?
            .batched(chunk_size)
            .await?;
        Ok(batched_reader)
    }

    #[cfg(feature = "async")]
    async fn num_rows_per_reader(&self, reader: &mut ParquetAsyncReader) -> PolarsResult<usize> {
        let predicate = self.predicate.clone();
        let num_rows;
        if predicate.is_some() {
            num_rows = reader.num_rows_with_predicate().await;
        } else {
            num_rows = reader.num_rows().await;
        }
        num_rows
    }

    #[allow(unused_variables)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        paths: Arc<Vec<PathBuf>>,
        options: ParquetOptions,
        cloud_options: Option<CloudOptions>,
        metadata: Option<FileMetaDataRef>,
        file_options: FileScanOptions,
        file_info: FileInfo,
        hive_parts: Option<Arc<Vec<HivePartitions>>>,
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
        let rows_left_to_read = file_options.slice.unwrap_or((0, usize::MAX)).1;

        let mut source = ParquetSource {
            batched_readers: VecDeque::new(),
            n_threads,
            processed_paths: 0,
            processed_rows: AtomicUsize::new(0),
            options,
            file_options,
            iter,
            paths,
            total_files_read: 0,
            cloud_options,
            metadata,
            file_info,
            hive_parts,
            verbose,
            run_async,
            prefetch_size,
            predicate,
            rows_left_to_read,
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
        if self.batched_readers.is_empty()
            && self.rows_left_to_read != 0
            && self.total_files_read != self.paths.len()
        {
            if self.run_async {
                let range = 0..self.prefetch_size - self.batched_readers.len();

                let range = range
                    .zip(&mut self.iter)
                    .map(|(_, index)| index)
                    .collect::<Vec<_>>();

                let readers = range
                    .clone()
                    .into_iter()
                    .map(|index| self.init_reader_async(index, self.rows_left_to_read));
                let mut readers = polars_io::pl_async::get_runtime()
                    .block_on(async { futures::future::try_join_all(readers).await })?;

                let num_rows_to_read = readers
                    .iter_mut()
                    .map(|(reader, _chunk_size)| self.num_rows_per_reader(reader));

                let num_rows_to_read = polars_io::pl_async::get_runtime()
                    .block_on(async { futures::future::try_join_all(num_rows_to_read).await })?;

                let num_rows_to_read = num_rows_to_read
                    .into_iter()
                    .zip(readers)
                    .map(|(rows_per_reader, (reader, chunk_size))| {
                        self.total_files_read += 1;
                        if self.rows_left_to_read == 0 {
                            return (reader, chunk_size, 0);
                        }
                        self.rows_left_to_read =
                            self.rows_left_to_read.saturating_sub(rows_per_reader);
                        (reader, chunk_size, rows_per_reader)
                    })
                    .filter(|(_reader, _chunk_size, rows_per_reader)| *rows_per_reader != 0)
                    .collect::<Vec<_>>();

                let init_iter =
                    num_rows_to_read
                        .into_iter()
                        .map(|(reader, chunk_size, _num_rows)| {
                            self.init_batch_reader(reader, chunk_size)
                        });

                let batched_readers =
                    polars_io::pl_async::get_runtime().block_on_potential_spawn(async {
                        futures::future::try_join_all(init_iter).await
                    })?;

                for r in batched_readers {
                    self.finish_init_reader(r)?;
                }
            } else {
                for _ in 0..self.prefetch_size - self.batched_readers.len() {
                    self.init_next_reader_sync()?
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
            if self.total_files_read != self.paths.len() && self.rows_left_to_read != 0 {
                return self.get_batches(_context);
            }
            return Ok(SourceResult::Finished);
        };

        let batches =
            get_runtime().block_on_potential_spawn(reader.next_batches(self.n_threads))?;

        Ok(match batches {
            None => {
                // reset the reader
                self.init_next_reader()?;
                return self.get_batches(_context);
            },
            Some(batches) => {
                let idx_offset = get_source_index(0);
                let out = batches
                    .into_iter()
                    .enumerate_u32()
                    .map(|(i, data)| DataChunk {
                        chunk_index: (idx_offset + i) as IdxSize,
                        data,
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
