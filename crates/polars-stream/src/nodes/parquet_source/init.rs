use std::future::Future;
use std::sync::Arc;

use futures::stream::FuturesUnordered;
use futures::StreamExt;
use polars_core::frame::DataFrame;
use polars_error::PolarsResult;
use polars_io::prelude::FileMetadata;
use polars_io::utils::byte_source::{DynByteSource, MemSliceByteSource};
use polars_io::utils::slice::SplitSlicePosition;
use polars_utils::mmap::MemSlice;
use polars_utils::pl_str::PlSmallStr;

use super::metadata_utils::{ensure_metadata_has_projected_fields, read_parquet_metadata_bytes};
use super::row_group_data_fetch::RowGroupDataFetcher;
use super::row_group_decode::RowGroupDecoder;
use super::{AsyncTaskData, ParquetSourceNode};
use crate::async_executor;
use crate::async_primitives::connector::connector;
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::morsel::get_ideal_morsel_size;
use crate::nodes::{MorselSeq, TaskPriority};
use crate::utils::task_handles_ext;

impl ParquetSourceNode {
    /// # Panics
    /// Panics if called more than once.
    async fn shutdown_impl(
        async_task_data: Arc<tokio::sync::Mutex<AsyncTaskData>>,
        verbose: bool,
    ) -> PolarsResult<()> {
        if verbose {
            eprintln!("[ParquetSource]: Shutting down");
        }

        let (mut raw_morsel_receivers, morsel_stream_task_handle) =
            async_task_data.try_lock().unwrap().take().unwrap();

        raw_morsel_receivers.clear();
        // Join on the producer handle to catch errors/panics.
        // Safety
        // * We dropped the receivers on the line above
        // * This function is only called once.
        morsel_stream_task_handle.await
    }

    pub(super) fn shutdown(&self) -> impl Future<Output = PolarsResult<()>> {
        if self.verbose {
            eprintln!("[ParquetSource]: Shutdown via `shutdown()`");
        }
        Self::shutdown_impl(self.async_task_data.clone(), self.verbose)
    }

    /// Spawns a task to shut down the source node to avoid blocking the current thread. This is
    /// usually called when data is no longer needed from the source node, as such it does not
    /// propagate any (non-critical) errors. If on the other hand the source node does not provide
    /// more data when requested, then it is more suitable to call [`Self::shutdown`], as it returns
    /// a result that can be used to distinguish between whether the data stream stopped due to an
    /// error or EOF.
    pub(super) fn shutdown_in_background(&self) {
        if self.verbose {
            eprintln!("[ParquetSource]: Shutdown via `shutdown_in_background()`");
        }
        let async_task_data = self.async_task_data.clone();
        polars_io::pl_async::get_runtime()
            .spawn(Self::shutdown_impl(async_task_data, self.verbose));
    }

    /// Constructs the task that provides a morsel stream.
    #[allow(clippy::type_complexity)]
    pub(super) fn init_raw_morsel_stream(
        &mut self,
    ) -> (
        Vec<crate::async_primitives::connector::Receiver<(DataFrame, MorselSeq, WaitToken)>>,
        async_executor::AbortOnDropHandle<PolarsResult<()>>,
    ) {
        let verbose = self.verbose;

        let use_statistics = self.options.use_statistics;

        let (mut raw_morsel_senders, raw_morsel_receivers): (Vec<_>, Vec<_>) =
            (0..self.config.num_pipelines).map(|_| connector()).unzip();

        if let Some((_, 0)) = self.file_options.slice {
            return (
                raw_morsel_receivers,
                async_executor::AbortOnDropHandle::new(async_executor::spawn(
                    TaskPriority::Low,
                    std::future::ready(Ok(())),
                )),
            );
        }

        let reader_schema = self
            .file_info
            .reader_schema
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap_left()
            .clone();

        let (normalized_slice_oneshot_rx, metadata_rx, metadata_task_handle) =
            self.init_metadata_fetcher();

        let num_pipelines = self.config.num_pipelines;
        let row_group_prefetch_size = self.config.row_group_prefetch_size;
        let projection = self.file_options.with_columns.clone();
        assert_eq!(self.physical_predicate.is_some(), self.predicate.is_some());
        let predicate = self.physical_predicate.clone();
        let memory_prefetch_func = self.memory_prefetch_func;

        let mut row_group_data_fetcher = RowGroupDataFetcher {
            metadata_rx,
            use_statistics,
            verbose,
            reader_schema,
            projection,
            predicate,
            slice_range: None, // Initialized later
            memory_prefetch_func,
            current_path_index: 0,
            current_byte_source: Default::default(),
            current_row_groups: Default::default(),
            current_row_group_idx: 0,
            current_max_row_group_height: 0,
            current_row_offset: 0,
            current_shared_file_state: Default::default(),
        };

        let row_group_decoder = self.init_row_group_decoder();
        let row_group_decoder = Arc::new(row_group_decoder);

        // Processes row group metadata and spawns I/O tasks to fetch row group data. This is
        // currently spawned onto the CPU runtime as it does not directly make any async I/O calls,
        // but instead it potentially performs predicate/slice evaluation on metadata. If we observe
        // that under heavy CPU load scenarios the I/O throughput drops due to this task not being
        // scheduled we can change it to be a high priority task.
        let morsel_stream_task_handle = async_executor::spawn(TaskPriority::Low, async move {
            let slice_range = {
                let Ok(slice) = normalized_slice_oneshot_rx.await else {
                    // If we are here then the producer probably errored.
                    drop(row_group_data_fetcher);
                    return metadata_task_handle.await.unwrap();
                };

                slice.map(|(offset, len)| offset..offset + len)
            };

            row_group_data_fetcher.slice_range = slice_range;

            // Pins a wait group to a channel index.
            struct IndexedWaitGroup {
                index: usize,
                wait_group: WaitGroup,
            }

            impl IndexedWaitGroup {
                async fn wait(self) -> Self {
                    self.wait_group.wait().await;
                    self
                }
            }

            // Ensure proper backpressure by only polling the buffered iterator when a wait group
            // is free.
            let mut wait_groups = (0..num_pipelines)
                .map(|index| {
                    let wait_group = WaitGroup::default();
                    {
                        let _prime_this_wait_group = wait_group.token();
                    }
                    IndexedWaitGroup {
                        index,
                        wait_group: WaitGroup::default(),
                    }
                    .wait()
                })
                .collect::<FuturesUnordered<_>>();

            let mut df_stream = row_group_data_fetcher
                .into_stream()
                .map(|x| async {
                    match x {
                        Ok(handle) => handle.await,
                        Err(e) => Err(e),
                    }
                })
                .buffered(row_group_prefetch_size)
                .map(|x| async {
                    let row_group_decoder = row_group_decoder.clone();

                    match x {
                        Ok(row_group_data) => {
                            async_executor::spawn(TaskPriority::Low, async move {
                                row_group_decoder.row_group_data_to_df(row_group_data).await
                            })
                            .await
                        },
                        Err(e) => Err(e),
                    }
                })
                .buffered(
                    // Because we are using an ordered buffer, we may suffer from head-of-line blocking,
                    // so we add a small amount of buffer.
                    num_pipelines + 4,
                );

            let morsel_seq_ref = &mut MorselSeq::default();
            let mut dfs = vec![].into_iter();

            'main: loop {
                let Some(mut indexed_wait_group) = wait_groups.next().await else {
                    break;
                };

                if dfs.len() == 0 {
                    let Some(v) = df_stream.next().await else {
                        break;
                    };

                    let v = v?;
                    assert!(!v.is_empty());

                    dfs = v.into_iter();
                }

                let mut df = dfs.next().unwrap();
                let morsel_seq = *morsel_seq_ref;
                *morsel_seq_ref = morsel_seq.successor();

                loop {
                    use crate::async_primitives::connector::SendError;

                    let channel_index = indexed_wait_group.index;
                    let wait_token = indexed_wait_group.wait_group.token();

                    match raw_morsel_senders[channel_index].try_send((df, morsel_seq, wait_token)) {
                        Ok(_) => {
                            wait_groups.push(indexed_wait_group.wait());
                            break;
                        },
                        Err(SendError::Closed(v)) => {
                            // The channel assigned to this wait group has been closed, so we will not
                            // add it back to the list of wait groups, and we will try to send this
                            // across another channel.
                            df = v.0
                        },
                        Err(SendError::Full(_)) => unreachable!(),
                    }

                    let Some(v) = wait_groups.next().await else {
                        // All channels have closed
                        break 'main;
                    };

                    indexed_wait_group = v;
                }
            }

            // Join on the producer handle to catch errors/panics.
            drop(df_stream);
            metadata_task_handle.await.unwrap()
        });

        let morsel_stream_task_handle =
            async_executor::AbortOnDropHandle::new(morsel_stream_task_handle);

        (raw_morsel_receivers, morsel_stream_task_handle)
    }

    /// Constructs the task that fetches file metadata.
    /// Note: This must be called AFTER `self.projected_arrow_fields` has been initialized.
    ///
    /// TODO: During IR conversion the metadata of the first file is already downloaded - see if
    /// we can find a way to re-use it.
    #[allow(clippy::type_complexity)]
    fn init_metadata_fetcher(
        &mut self,
    ) -> (
        tokio::sync::oneshot::Receiver<Option<(usize, usize)>>,
        crate::async_primitives::connector::Receiver<(
            usize,
            usize,
            Arc<DynByteSource>,
            FileMetadata,
            usize,
        )>,
        task_handles_ext::AbortOnDropHandle<PolarsResult<()>>,
    ) {
        let verbose = self.verbose;
        let io_runtime = polars_io::pl_async::get_runtime();

        assert!(
            !self.projected_arrow_fields.is_empty()
                || self.file_options.with_columns.as_deref() == Some(&[])
        );
        let projected_arrow_fields = self.projected_arrow_fields.clone();
        let needs_max_row_group_height_calc =
            self.file_options.include_file_paths.is_some() || self.hive_parts.is_some();

        let (normalized_slice_oneshot_tx, normalized_slice_oneshot_rx) =
            tokio::sync::oneshot::channel();
        let (mut metadata_tx, metadata_rx) = connector();

        let byte_source_builder = self.byte_source_builder.clone();

        if self.verbose {
            eprintln!(
                "[ParquetSource]: Byte source builder: {:?}",
                &byte_source_builder
            );
        }

        let fetch_metadata_bytes_for_path_index = {
            let scan_sources = &self.scan_sources;
            let cloud_options = Arc::new(self.cloud_options.clone());

            let scan_sources = scan_sources.clone();
            let cloud_options = cloud_options.clone();
            let byte_source_builder = byte_source_builder.clone();

            move |path_idx: usize| {
                let scan_sources = scan_sources.clone();
                let cloud_options = cloud_options.clone();
                let byte_source_builder = byte_source_builder.clone();

                let handle = io_runtime.spawn(async move {
                    let mut byte_source = Arc::new(
                        scan_sources
                            .get(path_idx)
                            .unwrap()
                            .to_dyn_byte_source(
                                &byte_source_builder,
                                cloud_options.as_ref().as_ref(),
                            )
                            .await?,
                    );
                    let (metadata_bytes, maybe_full_bytes) =
                        read_parquet_metadata_bytes(byte_source.as_ref(), verbose).await?;

                    if let Some(v) = maybe_full_bytes {
                        if !matches!(byte_source.as_ref(), DynByteSource::MemSlice(_)) {
                            if verbose {
                                eprintln!(
                                    "[ParquetSource]: Parquet file was fully fetched during \
                                         metadata read ({} bytes).",
                                    v.len(),
                                );
                            }

                            byte_source = Arc::new(DynByteSource::from(MemSliceByteSource(v)))
                        }
                    }

                    PolarsResult::Ok((path_idx, byte_source, metadata_bytes))
                });

                let handle = task_handles_ext::AbortOnDropHandle(handle);

                std::future::ready(handle)
            }
        };

        let process_metadata_bytes = {
            move |handle: task_handles_ext::AbortOnDropHandle<
                PolarsResult<(usize, Arc<DynByteSource>, MemSlice)>,
            >| {
                let projected_arrow_fields = projected_arrow_fields.clone();
                // Run on CPU runtime - metadata deserialization is expensive, especially
                // for very wide tables.
                let handle = async_executor::spawn(TaskPriority::Low, async move {
                    let (path_index, byte_source, metadata_bytes) = handle.await.unwrap()?;

                    let metadata = polars_parquet::parquet::read::deserialize_metadata(
                        metadata_bytes.as_ref(),
                        metadata_bytes.len() * 2 + 1024,
                    )?;

                    ensure_metadata_has_projected_fields(
                        projected_arrow_fields.as_ref(),
                        &metadata,
                    )?;

                    let file_max_row_group_height = if needs_max_row_group_height_calc {
                        metadata
                            .row_groups
                            .iter()
                            .map(|x| x.num_rows())
                            .max()
                            .unwrap_or(0)
                    } else {
                        0
                    };

                    PolarsResult::Ok((path_index, byte_source, metadata, file_max_row_group_height))
                });

                async_executor::AbortOnDropHandle::new(handle)
            }
        };

        let metadata_prefetch_size = self.config.metadata_prefetch_size;
        let metadata_decode_ahead_size = self.config.metadata_decode_ahead_size;

        let (start_tx, start_rx) = tokio::sync::oneshot::channel();
        self.morsel_stream_starter = Some(start_tx);

        let metadata_task_handle = if self
            .file_options
            .slice
            .map(|(offset, _)| offset >= 0)
            .unwrap_or(true)
        {
            normalized_slice_oneshot_tx
                .send(
                    self.file_options
                        .slice
                        .map(|(offset, len)| (offset as usize, len)),
                )
                .unwrap();

            // Safety: `offset + len` does not overflow.
            let slice_range = self
                .file_options
                .slice
                .map(|(offset, len)| offset as usize..offset as usize + len);

            let mut metadata_stream = futures::stream::iter(0..self.scan_sources.len())
                .map(fetch_metadata_bytes_for_path_index)
                .buffered(metadata_prefetch_size)
                .map(process_metadata_bytes)
                .buffered(metadata_decode_ahead_size);

            let scan_sources = self.scan_sources.clone();

            // We need to be able to both stop early as well as skip values, which is easier to do
            // using a custom task instead of futures::stream
            io_runtime.spawn(async move {
                let current_row_offset_ref = &mut 0usize;
                let current_path_index_ref = &mut 0usize;

                if start_rx.await.is_err() {
                    return Ok(());
                }

                if verbose {
                    eprintln!("[ParquetSource]: Starting data fetch")
                }

                loop {
                    let current_path_index = *current_path_index_ref;
                    *current_path_index_ref += 1;

                    let Some(v) = metadata_stream.next().await else {
                        break;
                    };

                    let (path_index, byte_source, metadata, file_max_row_group_height) = v
                        .map_err(|err| {
                            err.wrap_msg(|msg| {
                                format!(
                                    "error at path (index: {}, path: {:?}): {}",
                                    current_path_index,
                                    scan_sources
                                        .get(current_path_index)
                                        .map(|x| PlSmallStr::from_str(x.to_include_path_name())),
                                    msg
                                )
                            })
                        })?;

                    assert_eq!(path_index, current_path_index);

                    let current_row_offset = *current_row_offset_ref;
                    *current_row_offset_ref = current_row_offset.saturating_add(metadata.num_rows);

                    if let Some(slice_range) = slice_range.clone() {
                        match SplitSlicePosition::split_slice_at_file(
                            current_row_offset,
                            metadata.num_rows,
                            slice_range,
                        ) {
                            SplitSlicePosition::Before => {
                                if verbose {
                                    eprintln!(
                                        "[ParquetSource]: Slice pushdown: \
                                            Skipped file at index {} ({} rows)",
                                        current_path_index, metadata.num_rows
                                    );
                                }
                                continue;
                            },
                            SplitSlicePosition::After => unreachable!(),
                            SplitSlicePosition::Overlapping(..) => {},
                        };
                    };

                    if metadata_tx
                        .send((
                            path_index,
                            current_row_offset,
                            byte_source,
                            metadata,
                            file_max_row_group_height,
                        ))
                        .await
                        .is_err()
                    {
                        break;
                    }

                    if let Some(slice_range) = slice_range.as_ref() {
                        if *current_row_offset_ref >= slice_range.end {
                            if verbose {
                                eprintln!(
                                    "[ParquetSource]: Slice pushdown: \
                                        Stopped reading at file at index {} \
                                        (remaining {} files will not be read)",
                                    current_path_index,
                                    scan_sources.len() - current_path_index - 1,
                                );
                            }
                            break;
                        }
                    };
                }

                Ok(())
            })
        } else {
            // Walk the files in reverse to translate the slice into a positive offset.
            let slice = self.file_options.slice.unwrap();
            let slice_start_as_n_from_end = -slice.0 as usize;

            let mut metadata_stream = futures::stream::iter((0..self.scan_sources.len()).rev())
                .map(fetch_metadata_bytes_for_path_index)
                .buffered(metadata_prefetch_size)
                .map(process_metadata_bytes)
                .buffered(metadata_decode_ahead_size);

            // Note:
            // * We want to wait until the first morsel is requested before starting this
            let init_negative_slice_and_metadata = async move {
                let mut processed_metadata_rev = vec![];
                let mut cum_rows = 0;

                while let Some(v) = metadata_stream.next().await {
                    let v = v?;
                    let (_, _, metadata, _) = &v;
                    cum_rows += metadata.num_rows;
                    processed_metadata_rev.push(v);

                    if cum_rows >= slice_start_as_n_from_end {
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

                if len == 0 {
                    processed_metadata_rev.clear();
                }

                normalized_slice_oneshot_tx
                    .send(Some((start, len)))
                    .unwrap();

                let slice_range = start..(start + len);

                PolarsResult::Ok((slice_range, processed_metadata_rev, cum_rows))
            };

            let path_count = self.scan_sources.len();

            io_runtime.spawn(async move {
                if start_rx.await.is_err() {
                    return Ok(());
                }

                if verbose {
                    eprintln!("[ParquetSource]: Starting data fetch (negative slice)")
                }

                let (slice_range, processed_metadata_rev, cum_rows) =
                    async_executor::AbortOnDropHandle::new(async_executor::spawn(
                        TaskPriority::Low,
                        init_negative_slice_and_metadata,
                    ))
                    .await?;

                if verbose {
                    if let Some((path_index, ..)) = processed_metadata_rev.last() {
                        eprintln!(
                            "[ParquetSource]: Slice pushdown: Negatively-offsetted slice {:?} \
                            begins at file index {}, translated to {:?}",
                            slice, path_index, slice_range
                        );
                    } else {
                        eprintln!(
                            "[ParquetSource]: Slice pushdown: Negatively-offsetted slice {:?} \
                            skipped all files ({} files containing {} rows)",
                            slice, path_count, cum_rows
                        )
                    }
                }

                let metadata_iter = processed_metadata_rev.into_iter().rev();
                let current_row_offset_ref = &mut 0usize;

                for (current_path_index, byte_source, metadata, file_max_row_group_height) in
                    metadata_iter
                {
                    let current_row_offset = *current_row_offset_ref;
                    *current_row_offset_ref = current_row_offset.saturating_add(metadata.num_rows);

                    assert!(matches!(
                        SplitSlicePosition::split_slice_at_file(
                            current_row_offset,
                            metadata.num_rows,
                            slice_range.clone(),
                        ),
                        SplitSlicePosition::Overlapping(..)
                    ));

                    if metadata_tx
                        .send((
                            current_path_index,
                            current_row_offset,
                            byte_source,
                            metadata,
                            file_max_row_group_height,
                        ))
                        .await
                        .is_err()
                    {
                        break;
                    }

                    if *current_row_offset_ref >= slice_range.end {
                        if verbose {
                            eprintln!(
                                "[ParquetSource]: Slice pushdown: \
                                Stopped reading at file at index {} \
                                (remaining {} files will not be read)",
                                current_path_index,
                                path_count - current_path_index - 1,
                            );
                        }
                        break;
                    }
                }

                Ok(())
            })
        };

        let metadata_task_handle = task_handles_ext::AbortOnDropHandle(metadata_task_handle);

        (
            normalized_slice_oneshot_rx,
            metadata_rx,
            metadata_task_handle,
        )
    }

    /// Creates a `RowGroupDecoder` that turns `RowGroupData` into DataFrames.
    /// This must be called AFTER the following have been initialized:
    /// * `self.projected_arrow_fields`
    /// * `self.physical_predicate`
    pub(super) fn init_row_group_decoder(&self) -> RowGroupDecoder {
        assert!(
            !self.projected_arrow_fields.is_empty()
                || self.file_options.with_columns.as_deref() == Some(&[])
        );
        assert_eq!(self.predicate.is_some(), self.physical_predicate.is_some());

        let scan_sources = self.scan_sources.clone();
        let hive_partitions = self.hive_parts.clone();
        let hive_partitions_width = hive_partitions
            .as_deref()
            .map(|x| x[0].get_statistics().column_stats().len())
            .unwrap_or(0);
        let include_file_paths = self.file_options.include_file_paths.clone();
        let projected_arrow_fields = self.projected_arrow_fields.clone();
        let row_index = self.file_options.row_index.clone();
        let physical_predicate = self.physical_predicate.clone();
        let ideal_morsel_size = get_ideal_morsel_size();

        RowGroupDecoder {
            scan_sources,
            hive_partitions,
            hive_partitions_width,
            include_file_paths,
            projected_arrow_fields,
            row_index,
            physical_predicate,
            ideal_morsel_size,
        }
    }

    pub(super) fn init_projected_arrow_fields(&mut self) {
        let reader_schema = self
            .file_info
            .reader_schema
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap_left()
            .clone();

        self.projected_arrow_fields =
            if let Some(columns) = self.file_options.with_columns.as_deref() {
                columns
                    .iter()
                    .map(|x| reader_schema.get(x).unwrap().clone())
                    .collect()
            } else {
                reader_schema.iter_values().cloned().collect()
            };

        if self.verbose {
            eprintln!(
                "[ParquetSource]: {} columns to be projected from {} files",
                self.projected_arrow_fields.len(),
                self.scan_sources.len(),
            );
        }
    }
}
