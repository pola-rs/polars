use hive::HivePartitions;
use polars_core::config;
#[cfg(feature = "cloud")]
use polars_core::config::{get_file_prefetch_size, verbose};
use polars_core::utils::accumulate_dataframes_vertical;
use polars_error::feature_gated;
use polars_io::cloud::CloudOptions;
use polars_io::parquet::metadata::FileMetadataRef;
use polars_io::utils::slice::split_slice_at_file;
use polars_io::RowIndex;

use super::*;

pub struct ParquetExec {
    sources: ScanSources,
    file_info: FileInfo,

    hive_parts: Option<Arc<Vec<HivePartitions>>>,

    predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) options: ParquetOptions,
    #[allow(dead_code)]
    cloud_options: Option<CloudOptions>,
    file_options: FileScanOptions,
    #[allow(dead_code)]
    metadata: Option<FileMetadataRef>,
}

impl ParquetExec {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        sources: ScanSources,
        file_info: FileInfo,
        hive_parts: Option<Arc<Vec<HivePartitions>>>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        options: ParquetOptions,
        cloud_options: Option<CloudOptions>,
        file_options: FileScanOptions,
        metadata: Option<FileMetadataRef>,
    ) -> Self {
        ParquetExec {
            sources,
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
            ParallelStrategy::Auto if self.sources.len() > POOL.current_num_threads() => {
                ParallelStrategy::RowGroups
            },
            identity => identity,
        };

        let mut result = vec![];

        let step = std::cmp::min(POOL.current_num_threads(), 128);
        // Modified if we have a negative slice
        let mut first_source = 0;

        let first_schema = self.file_info.reader_schema.clone().unwrap().unwrap_left();

        let projected_arrow_schema = {
            if let Some(with_columns) = self.file_options.with_columns.as_deref() {
                Some(Arc::new(first_schema.try_project(with_columns)?))
            } else {
                None
            }
        };
        let predicate = self.predicate.clone().map(phys_expr_to_io_expr);
        let mut base_row_index = self.file_options.row_index.take();

        // (offset, end)
        let (slice_offset, slice_end) = if let Some(slice) = self.file_options.slice {
            if slice.0 >= 0 {
                (slice.0 as usize, slice.1.saturating_add(slice.0 as usize))
            } else {
                // Walk the files in reverse until we find the first file, and then translate the
                // slice into a positive-offset equivalent.
                let slice_start_as_n_from_end = -slice.0 as usize;
                let mut cum_rows = 0;
                let mut first_source_row_offset = 0;
                let chunk_size = 8;
                POOL.install(|| {
                    for path_indexes in (0..self.sources.len())
                        .rev()
                        .collect::<Vec<_>>()
                        .chunks(chunk_size)
                    {
                        let row_counts = path_indexes
                            .into_par_iter()
                            .map(|&i| {
                                let memslice = self.sources.at(i).to_memslice()?;

                                let mut reader = ParquetReader::new(std::io::Cursor::new(memslice));

                                if i == 0 {
                                    if let Some(md) = self.metadata.clone() {
                                        reader.set_metadata(md)
                                    }
                                }

                                reader.num_rows()
                            })
                            .collect::<PolarsResult<Vec<_>>>()?;

                        for (path_idx, rc) in path_indexes.iter().zip(row_counts) {
                            if first_source == 0 {
                                cum_rows += rc;

                                if cum_rows >= slice_start_as_n_from_end {
                                    first_source = *path_idx;

                                    if base_row_index.is_none() {
                                        break;
                                    }
                                }
                            } else {
                                first_source_row_offset += rc;
                            }
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

                if let Some(ri) = base_row_index.as_mut() {
                    ri.offset += first_source_row_offset as IdxSize;
                }

                (start, end)
            }
        } else {
            (0, usize::MAX)
        };

        let mut current_offset = 0;
        // Limit no. of files at a time to prevent open file limits.

        for i in (first_source..self.sources.len()).step_by(step) {
            let end = std::cmp::min(i.saturating_add(step), self.sources.len());

            if current_offset >= slice_end && !result.is_empty() {
                return Ok(result);
            }

            // First initialize the readers, predicates and metadata.
            // This will be used to determine the slices. That way we can actually read all the
            // files in parallel even if we add row index columns or slices.
            let iter = (i..end).into_par_iter().map(|i| {
                let source = self.sources.at(i);
                let hive_partitions = self
                    .hive_parts
                    .as_ref()
                    .map(|x| x[i].materialize_partition_columns());

                let memslice = source.to_memslice()?;

                let mut reader = ParquetReader::new(std::io::Cursor::new(memslice));

                if i == 0 {
                    if let Some(md) = self.metadata.clone() {
                        reader.set_metadata(md)
                    }
                }

                let mut reader = reader
                    .read_parallel(parallel)
                    .set_low_memory(self.options.low_memory)
                    .use_statistics(self.options.use_statistics)
                    .set_rechunk(false)
                    .with_hive_partition_columns(hive_partitions)
                    .with_include_file_path(
                        self.file_options
                            .include_file_paths
                            .as_ref()
                            .map(|x| (x.clone(), Arc::from(source.to_include_path_name()))),
                    );

                reader.num_rows().map(|num_rows| (reader, num_rows))
            });

            // We do this in parallel because wide tables can take a long time deserializing metadata.
            let readers_and_metadata = POOL.install(|| iter.collect::<PolarsResult<Vec<_>>>())?;

            let current_offset_ref = &mut current_offset;
            let row_statistics = readers_and_metadata
                .iter()
                .map(|(_, num_rows)| {
                    let cum_rows = *current_offset_ref;
                    (
                        cum_rows,
                        split_slice_at_file(current_offset_ref, *num_rows, slice_offset, slice_end),
                    )
                })
                .collect::<Vec<_>>();

            let allow_missing_columns = self.file_options.allow_missing_columns;

            let out = POOL.install(|| {
                readers_and_metadata
                    .into_par_iter()
                    .zip(row_statistics.into_par_iter())
                    .map(|((reader, _), (cumulative_read, slice))| {
                        let row_index = base_row_index.as_ref().map(|rc| RowIndex {
                            name: rc.name.clone(),
                            offset: rc.offset + cumulative_read as IdxSize,
                        });

                        let df = reader
                            .with_slice(Some(slice))
                            .with_row_index(row_index)
                            .with_predicate(predicate.clone())
                            .with_arrow_schema_projection(
                                &first_schema,
                                projected_arrow_schema.as_deref(),
                                allow_missing_columns,
                            )?
                            .finish()?;

                        Ok(df)
                    })
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
        let paths = self.sources.into_paths().unwrap();
        let first_metadata = &self.metadata;
        let cloud_options = self.cloud_options.as_ref();

        let mut result = vec![];
        let batch_size = get_file_prefetch_size();

        if verbose {
            eprintln!("POLARS PREFETCH_SIZE: {}", batch_size)
        }

        let first_schema = self.file_info.reader_schema.clone().unwrap().unwrap_left();

        let projected_arrow_schema = {
            if let Some(with_columns) = self.file_options.with_columns.as_deref() {
                Some(Arc::new(first_schema.try_project(with_columns)?))
            } else {
                None
            }
        };
        let predicate = self.predicate.clone().map(phys_expr_to_io_expr);
        let mut base_row_index = self.file_options.row_index.take();

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
                let mut first_source_row_offset = 0;

                let paths = &paths;
                let cloud_options = Arc::new(self.cloud_options.clone());

                let paths = paths.clone();
                let cloud_options = cloud_options.clone();

                let mut iter = stream::iter((0..paths.len()).rev().map(|i| {
                    let paths = paths.clone();
                    let cloud_options = cloud_options.clone();
                    let first_metadata = first_metadata.clone();

                    pl_async::get_runtime().spawn(async move {
                        PolarsResult::Ok((
                            i,
                            ParquetAsyncReader::from_uri(
                                paths[i].to_str().unwrap(),
                                cloud_options.as_ref().as_ref(),
                                first_metadata.filter(|_| i == 0),
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

                    if first_file_idx == 0 {
                        cum_rows += num_rows;

                        if cum_rows >= slice_start_as_n_from_end {
                            first_file_idx = path_idx;

                            if base_row_index.is_none() {
                                break;
                            }
                        }
                    } else {
                        first_source_row_offset += num_rows;
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

                if let Some(ri) = base_row_index.as_mut() {
                    ri.offset += first_source_row_offset as IdxSize;
                }

                (start, end)
            }
        } else {
            (0, usize::MAX)
        };

        let mut current_offset = 0;
        let mut processed = 0;

        for batch_start in (first_file_idx..paths.len()).step_by(batch_size) {
            let end = std::cmp::min(batch_start.saturating_add(batch_size), paths.len());
            let paths = &paths[batch_start..end];
            let hive_parts = self.hive_parts.as_ref().map(|x| &x[batch_start..end]);

            if current_offset >= slice_end && !result.is_empty() {
                return Ok(result);
            }
            processed += paths.len();
            if verbose {
                eprintln!(
                    "querying metadata of {}/{} files...",
                    processed,
                    paths.len()
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
            let use_statistics = self.options.use_statistics;
            let base_row_index_ref = &base_row_index;
            let include_file_paths = self.file_options.include_file_paths.as_ref();
            let first_schema = first_schema.clone();
            let projected_arrow_schema = projected_arrow_schema.clone();
            let predicate = predicate.clone();
            let allow_missing_columns = self.file_options.allow_missing_columns;

            if verbose {
                eprintln!("reading of {}/{} file...", processed, paths.len());
            }

            let iter = readers_and_metadata
                .into_iter()
                .enumerate()
                .map(|(i, (_, reader))| {
                    let first_schema = first_schema.clone();
                    let projected_arrow_schema = projected_arrow_schema.clone();
                    let predicate = predicate.clone();
                    let (cumulative_read, slice) = row_statistics[i];
                    let hive_partitions = hive_parts
                        .as_ref()
                        .map(|x| x[i].materialize_partition_columns());

                    async move {
                        let row_index = base_row_index_ref.as_ref().map(|rc| RowIndex {
                            name: rc.name.clone(),
                            offset: rc.offset + cumulative_read as IdxSize,
                        });

                        let df = reader
                            .with_slice(Some(slice))
                            .with_row_index(row_index)
                            .with_arrow_schema_projection(
                                &first_schema,
                                projected_arrow_schema.as_deref(),
                                allow_missing_columns,
                            )
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

    fn read_impl(&mut self) -> PolarsResult<DataFrame> {
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

        let is_cloud = self.sources.is_cloud_url();
        let force_async = config::force_async();

        let out = if is_cloud || (self.sources.is_paths() && force_async) {
            feature_gated!("cloud", {
                if force_async && config::verbose() {
                    eprintln!("ASYNC READING FORCED");
                }

                polars_io::pl_async::get_runtime().block_on_potential_spawn(self.read_async())?
            })
        } else {
            self.read_par()?
        };

        let mut out = accumulate_dataframes_vertical(out)?;

        let num_unfiltered_rows = out.height();
        self.file_info.row_estimation = (Some(num_unfiltered_rows), num_unfiltered_rows);

        polars_io::predicates::apply_predicate(&mut out, post_predicate.as_deref(), true)?;

        if self.file_options.rechunk {
            out.as_single_chunk_par();
        }
        Ok(out)
    }

    fn metadata_sync(&mut self) -> PolarsResult<&FileMetadataRef> {
        let memslice = self.sources.get(0).unwrap().to_memslice()?;
        Ok(self.metadata.insert(
            ParquetReader::new(std::io::Cursor::new(memslice))
                .get_metadata()?
                .clone(),
        ))
    }

    #[cfg(feature = "cloud")]
    async fn metadata_async(&mut self) -> PolarsResult<&FileMetadataRef> {
        let ScanSourceRef::Path(path) = self.sources.get(0).unwrap() else {
            unreachable!();
        };

        let mut reader =
            ParquetAsyncReader::from_uri(path.to_str().unwrap(), self.cloud_options.as_ref(), None)
                .await?;

        Ok(self.metadata.insert(reader.get_metadata().await?.clone()))
    }

    fn metadata(&mut self) -> PolarsResult<&FileMetadataRef> {
        let metadata = self.metadata.take();
        if let Some(md) = metadata {
            return Ok(self.metadata.insert(md));
        }

        #[cfg(feature = "cloud")]
        if self.sources.is_cloud_url() {
            return polars_io::pl_async::get_runtime()
                .block_on_potential_spawn(self.metadata_async());
        }

        self.metadata_sync()
    }
}

impl ScanExec for ParquetExec {
    fn read(
        &mut self,
        with_columns: Option<Arc<[PlSmallStr]>>,
        slice: Option<(usize, usize)>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        row_index: Option<RowIndex>,
    ) -> PolarsResult<DataFrame> {
        self.file_options.with_columns = with_columns;
        self.file_options.slice = slice.map(|(o, l)| (o as i64, l));
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

        let md = self.metadata()?;
        let arrow_schema = polars_io::parquet::read::infer_schema(md)?;
        self.file_info.schema =
            Arc::new(Schema::from_iter(arrow_schema.iter().map(
                |(name, field)| (name.clone(), DataType::from_arrow_field(field)),
            )));
        self.file_info.reader_schema = Some(arrow::Either::Left(Arc::new(arrow_schema)));

        Ok(&self.file_info.schema)
    }

    fn num_unfiltered_rows(&mut self) -> PolarsResult<IdxSize> {
        let md = self.metadata()?;
        Ok(md.num_rows as IdxSize)
    }
}

impl Executor for ParquetExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let profile_name = if state.has_node_timer() {
            let mut ids = vec![self.sources.id()];
            if self.predicate.is_some() {
                ids.push("predicate".into())
            }
            let name = comma_delimited("parquet".to_string(), &ids);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(|| self.read_impl(), profile_name)
    }
}
