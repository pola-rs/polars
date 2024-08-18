use std::path::PathBuf;

use hive::HivePartitions;
use polars_core::config;
#[cfg(feature = "cloud")]
use polars_core::config::{get_file_prefetch_size, verbose};
use polars_core::utils::accumulate_dataframes_vertical;
use polars_io::cloud::CloudOptions;
use polars_io::parquet::metadata::FileMetaDataRef;
use polars_io::path_utils::is_cloud_url;
use polars_io::utils::slice::split_slice_at_file;
use polars_io::RowIndex;

use super::*;

pub struct ParquetExec {
    paths: Arc<Vec<PathBuf>>,
    file_info: FileInfo,
    hive_parts: Option<Arc<Vec<HivePartitions>>>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    options: ParquetOptions,
    #[allow(dead_code)]
    cloud_options: Option<CloudOptions>,
    file_options: FileScanOptions,
    #[allow(dead_code)]
    metadata: Option<FileMetaDataRef>,
}

impl ParquetExec {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        paths: Arc<Vec<PathBuf>>,
        file_info: FileInfo,
        hive_parts: Option<Arc<Vec<HivePartitions>>>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        options: ParquetOptions,
        cloud_options: Option<CloudOptions>,
        file_options: FileScanOptions,
        metadata: Option<FileMetaDataRef>,
    ) -> Self {
        ParquetExec {
            paths,
            file_info,
            hive_parts,
            predicate,
            options,
            cloud_options,
            file_options,
            metadata,
        }
    }

    fn read_par(&mut self) -> PolarsResult<Vec<DataFrame>> {
        let parallel = match self.options.parallel {
            ParallelStrategy::Auto if self.paths.len() > POOL.current_num_threads() => {
                ParallelStrategy::RowGroups
            },
            identity => identity,
        };

        let mut result = vec![];

        let step = std::cmp::min(POOL.current_num_threads(), 128);
        // Modified if we have a negative slice
        let mut first_file = 0;

        // (offset, end)
        let (slice_offset, slice_end) = if let Some(slice) = self.file_options.slice {
            if slice.0 >= 0 {
                (slice.0 as usize, slice.1.saturating_add(slice.0 as usize))
            } else {
                // Walk the files in reverse until we find the first file, and then translate the
                // slice into a positive-offset equivalent.
                let slice_start_as_n_from_end = -slice.0 as usize;
                let mut cum_rows = 0;
                let chunk_size = 8;
                POOL.install(|| {
                    for path_indexes in (0..self.paths.len())
                        .rev()
                        .collect::<Vec<_>>()
                        .chunks(chunk_size)
                    {
                        let row_counts = path_indexes
                            .into_par_iter()
                            .map(|i| {
                                ParquetReader::new(std::fs::File::open(&self.paths[*i])?).num_rows()
                            })
                            .collect::<PolarsResult<Vec<_>>>()?;

                        for (path_idx, rc) in path_indexes.iter().zip(row_counts) {
                            cum_rows += rc;

                            if cum_rows >= slice_start_as_n_from_end {
                                first_file = *path_idx;
                                break;
                            }
                        }

                        if first_file > 0 {
                            break;
                        }
                    }

                    PolarsResult::Ok(())
                })?;

                let (start, len) = if slice_start_as_n_from_end > cum_rows {
                    // We need to trim the slice, e.g. SLICE[offset: -100, len: 75] on a file of 50
                    // rows should only give the first 25 rows.
                    let first_file_position = slice_start_as_n_from_end - cum_rows;
                    (0, slice.1.saturating_sub(first_file_position))
                } else {
                    (cum_rows - slice_start_as_n_from_end, slice.1)
                };

                let end = start.saturating_add(len);

                (start, end)
            }
        } else {
            (0, usize::MAX)
        };

        let mut current_offset = 0;
        let base_row_index = self.file_options.row_index.take();
        // Limit no. of files at a time to prevent open file limits.

        for i in (first_file..self.paths.len()).step_by(step) {
            let end = std::cmp::min(i.saturating_add(step), self.paths.len());
            let paths = &self.paths[i..end];
            let hive_parts = self.hive_parts.as_ref().map(|x| &x[i..end]);

            if current_offset >= slice_end && !result.is_empty() {
                return Ok(result);
            }

            // First initialize the readers, predicates and metadata.
            // This will be used to determine the slices. That way we can actually read all the
            // files in parallel even if we add row index columns or slices.
            let iter = (0..paths.len()).into_par_iter().map(|i| {
                let path = &paths[i];
                let hive_partitions = hive_parts.map(|x| x[i].materialize_partition_columns());

                let file = std::fs::File::open(path)?;
                let (projection, predicate) = prepare_scan_args(
                    self.predicate.clone(),
                    &mut self.file_options.with_columns.clone(),
                    &mut self.file_info.schema.clone(),
                    base_row_index.is_some(),
                    hive_partitions.as_deref(),
                );

                let mut reader = ParquetReader::new(file)
                    .read_parallel(parallel)
                    .set_low_memory(self.options.low_memory)
                    .use_statistics(self.options.use_statistics)
                    .set_rechunk(false)
                    .with_hive_partition_columns(hive_partitions)
                    .with_include_file_path(
                        self.file_options
                            .include_file_paths
                            .as_ref()
                            .map(|x| (x.clone(), Arc::from(paths[i].to_str().unwrap()))),
                    );

                reader
                    .num_rows()
                    .map(|num_rows| (reader, num_rows, predicate, projection))
            });

            // We do this in parallel because wide tables can take a long time deserializing metadata.
            let readers_and_metadata = POOL.install(|| iter.collect::<PolarsResult<Vec<_>>>())?;

            let current_offset_ref = &mut current_offset;
            let row_statistics = readers_and_metadata
                .iter()
                .map(|(_, num_rows, _, _)| {
                    let cum_rows = *current_offset_ref;
                    (
                        cum_rows,
                        split_slice_at_file(current_offset_ref, *num_rows, slice_offset, slice_end),
                    )
                })
                .collect::<Vec<_>>();

            let out = POOL.install(|| {
                readers_and_metadata
                    .into_par_iter()
                    .zip(row_statistics.into_par_iter())
                    .map(
                        |((reader, _, predicate, projection), (cumulative_read, slice))| {
                            let row_index = base_row_index.as_ref().map(|rc| RowIndex {
                                name: rc.name.clone(),
                                offset: rc.offset + cumulative_read as IdxSize,
                            });

                            let df = reader
                                .with_slice(Some(slice))
                                .with_row_index(row_index)
                                .with_predicate(predicate.clone())
                                .with_projection(projection.clone())
                                .check_schema(
                                    self.file_info
                                        .reader_schema
                                        .clone()
                                        .unwrap()
                                        .unwrap_left()
                                        .as_ref(),
                                )?
                                .finish()?;

                            Ok(df)
                        },
                    )
                    .collect::<PolarsResult<Vec<_>>>()
            })?;

            if result.is_empty() {
                result = out;
            } else {
                result.extend_from_slice(&out)
            }
        }
        Ok(result)
    }

    #[cfg(feature = "cloud")]
    async fn read_async(&mut self) -> PolarsResult<Vec<DataFrame>> {
        use futures::{stream, StreamExt};
        use polars_io::pl_async;
        use polars_io::utils::slice::split_slice_at_file;

        let verbose = verbose();
        let first_metadata = &self.metadata;
        let cloud_options = self.cloud_options.as_ref();

        let mut result = vec![];
        let batch_size = get_file_prefetch_size();

        if verbose {
            eprintln!("POLARS PREFETCH_SIZE: {}", batch_size)
        }

        // Modified if we have a negative slice
        let mut first_file_idx = 0;

        // (offset, end)
        let (slice_offset, slice_end) = if let Some(slice) = self.file_options.slice {
            if slice.0 >= 0 {
                (slice.0 as usize, slice.1.saturating_add(slice.0 as usize))
            } else {
                // Walk the files in reverse until we find the first file, and then translate the
                // slice into a positive-offset equivalent.
                let slice_start_as_n_from_end = -slice.0 as usize;
                let mut cum_rows = 0;

                let paths = &self.paths;
                let cloud_options = Arc::new(self.cloud_options.clone());

                let paths = paths.clone();
                let cloud_options = cloud_options.clone();

                let mut iter = stream::iter((0..self.paths.len()).rev().map(|i| {
                    let paths = paths.clone();
                    let cloud_options = cloud_options.clone();

                    pl_async::get_runtime().spawn(async move {
                        PolarsResult::Ok((
                            i,
                            ParquetAsyncReader::from_uri(
                                paths[i].to_str().unwrap(),
                                cloud_options.as_ref().as_ref(),
                                None,
                            )
                            .await?
                            .num_rows()
                            .await?,
                        ))
                    })
                }))
                .buffered(8);

                while let Some(v) = iter.next().await {
                    let (path_idx, num_rows) = v.unwrap()?;

                    cum_rows += num_rows;

                    if cum_rows >= slice_start_as_n_from_end {
                        first_file_idx = path_idx;
                        break;
                    }
                }

                let (start, len) = if slice_start_as_n_from_end > cum_rows {
                    // We need to trim the slice, e.g. SLICE[offset: -100, len: 75] on a file of 50
                    // rows should only give the first 25 rows.
                    let first_file_position = slice_start_as_n_from_end - cum_rows;
                    (0, slice.1.saturating_sub(first_file_position))
                } else {
                    (cum_rows - slice_start_as_n_from_end, slice.1)
                };

                let end = start.saturating_add(len);

                (start, end)
            }
        } else {
            (0, usize::MAX)
        };

        let mut current_offset = 0;
        let base_row_index = self.file_options.row_index.take();
        let mut processed = 0;

        for batch_start in (first_file_idx..self.paths.len()).step_by(batch_size) {
            let end = std::cmp::min(batch_start.saturating_add(batch_size), self.paths.len());
            let paths = &self.paths[batch_start..end];
            let hive_parts = self.hive_parts.as_ref().map(|x| &x[batch_start..end]);

            if current_offset >= slice_end && !result.is_empty() {
                return Ok(result);
            }
            processed += paths.len();
            if verbose {
                eprintln!(
                    "querying metadata of {}/{} files...",
                    processed,
                    self.paths.len()
                );
            }

            // First initialize the readers and get the metadata concurrently.
            let iter = paths.iter().enumerate().map(|(i, path)| async move {
                let first_file = batch_start == 0 && i == 0;
                // use the cached one as this saves a cloud call
                let metadata = if first_file {
                    first_metadata.clone()
                } else {
                    None
                };
                let mut reader =
                    ParquetAsyncReader::from_uri(&path.to_string_lossy(), cloud_options, metadata)
                        .await?;

                let num_rows = reader.num_rows().await?;
                PolarsResult::Ok((num_rows, reader))
            });
            let readers_and_metadata = futures::future::try_join_all(iter).await?;

            let current_offset_ref = &mut current_offset;

            // Then compute `n_rows` to be taken per file up front, so we can actually read concurrently
            // after this.
            let row_statistics = readers_and_metadata
                .iter()
                .map(|(num_rows, _)| {
                    let cum_rows = *current_offset_ref;
                    (
                        cum_rows,
                        split_slice_at_file(current_offset_ref, *num_rows, slice_offset, slice_end),
                    )
                })
                .collect::<Vec<_>>();

            // Now read the actual data.
            let file_info = &self.file_info;
            let file_options = &self.file_options;
            let use_statistics = self.options.use_statistics;
            let predicate = &self.predicate;
            let base_row_index_ref = &base_row_index;
            let include_file_paths = self.file_options.include_file_paths.as_ref();

            if verbose {
                eprintln!("reading of {}/{} file...", processed, self.paths.len());
            }

            let iter = readers_and_metadata
                .into_iter()
                .enumerate()
                .map(|(i, (_, reader))| {
                    let (cumulative_read, slice) = row_statistics[i];
                    let hive_partitions = hive_parts
                        .as_ref()
                        .map(|x| x[i].materialize_partition_columns());

                    let schema = self
                        .file_info
                        .reader_schema
                        .as_ref()
                        .unwrap()
                        .as_ref()
                        .unwrap_left()
                        .clone();

                    async move {
                        let file_info = file_info.clone();
                        let row_index = base_row_index_ref.as_ref().map(|rc| RowIndex {
                            name: rc.name.clone(),
                            offset: rc.offset + cumulative_read as IdxSize,
                        });

                        let (projection, predicate) = prepare_scan_args(
                            predicate.clone(),
                            &mut file_options.with_columns.clone(),
                            &mut file_info.schema.clone(),
                            row_index.is_some(),
                            hive_partitions.as_deref(),
                        );

                        let df = reader
                            .with_slice(Some(slice))
                            .with_row_index(row_index)
                            .with_projection(projection)
                            .check_schema(schema.as_ref())
                            .await?
                            .use_statistics(use_statistics)
                            .with_predicate(predicate)
                            .set_rechunk(false)
                            .with_hive_partition_columns(hive_partitions)
                            .with_include_file_path(
                                include_file_paths
                                    .map(|x| (x.clone(), Arc::from(paths[i].to_str().unwrap()))),
                            )
                            .finish()
                            .await?;

                        PolarsResult::Ok(df)
                    }
                });

            let dfs = futures::future::try_join_all(iter).await?;
            result.extend(dfs.into_iter())
        }

        Ok(result)
    }

    fn read(&mut self) -> PolarsResult<DataFrame> {
        // FIXME: The row index implementation is incorrect when a predicate is
        // applied. This code mitigates that by applying the predicate after the
        // collection of the entire dataframe if a row index is requested. This is
        // inefficient.
        let post_predicate = self
            .file_options
            .row_index
            .as_ref()
            .and_then(|_| self.predicate.take())
            .map(phys_expr_to_io_expr);

        let is_cloud = is_cloud_url(self.paths.first().unwrap());
        let force_async = config::force_async();

        let out = if is_cloud || force_async {
            #[cfg(not(feature = "cloud"))]
            {
                panic!("activate cloud feature")
            }

            #[cfg(feature = "cloud")]
            {
                if force_async && config::verbose() {
                    eprintln!("ASYNC READING FORCED");
                }

                polars_io::pl_async::get_runtime().block_on_potential_spawn(self.read_async())?
            }
        } else {
            self.read_par()?
        };

        let mut out = accumulate_dataframes_vertical(out)?;

        polars_io::predicates::apply_predicate(&mut out, post_predicate.as_deref(), true)?;

        if self.file_options.rechunk {
            out.as_single_chunk_par();
        }
        Ok(out)
    }
}

impl Executor for ParquetExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let profile_name = if state.has_node_timer() {
            let mut ids = vec![self.paths[0].to_string_lossy().into()];
            if self.predicate.is_some() {
                ids.push("predicate".into())
            }
            let name = comma_delimited("parquet".to_string(), &ids);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(|| self.read(), profile_name)
    }
}
