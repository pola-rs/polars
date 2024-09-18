use std::future::Future;
use std::sync::Arc;

use futures::stream::FuturesUnordered;
use futures::StreamExt;
use polars_core::frame::DataFrame;
use polars_error::PolarsResult;
use polars_io::prelude::ParallelStrategy;
use polars_io::prelude::_internal::PrefilterMaskSetting;

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
        morsel_stream_task_handle.await.unwrap()
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

    /// Constructs the task that distributes morsels across the engine pipelines.
    #[allow(clippy::type_complexity)]
    pub(super) fn init_raw_morsel_distributor(
        &mut self,
    ) -> (
        Vec<crate::async_primitives::connector::Receiver<(DataFrame, MorselSeq, WaitToken)>>,
        task_handles_ext::AbortOnDropHandle<PolarsResult<()>>,
    ) {
        let verbose = self.verbose;
        let io_runtime = polars_io::pl_async::get_runtime();

        let use_statistics = self.options.use_statistics;

        let (mut raw_morsel_senders, raw_morsel_receivers): (Vec<_>, Vec<_>) =
            (0..self.config.num_pipelines).map(|_| connector()).unzip();

        if let Some((_, 0)) = self.file_options.slice {
            return (
                raw_morsel_receivers,
                task_handles_ext::AbortOnDropHandle(io_runtime.spawn(std::future::ready(Ok(())))),
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

        // Distributes morsels across pipelines. This does not perform any CPU or I/O bound work -
        // it is purely a dispatch loop.
        let raw_morsel_distributor_task_handle = io_runtime.spawn(async move {
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
                        Ok(handle) => handle.await.unwrap(),
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

        let raw_morsel_distributor_task_handle =
            task_handles_ext::AbortOnDropHandle(raw_morsel_distributor_task_handle);

        (raw_morsel_receivers, raw_morsel_distributor_task_handle)
    }

    /// Creates a `RowGroupDecoder` that turns `RowGroupData` into DataFrames.
    /// This must be called AFTER the following have been initialized:
    /// * `self.projected_arrow_schema`
    /// * `self.physical_predicate`
    pub(super) fn init_row_group_decoder(&self) -> RowGroupDecoder {
        assert_eq!(self.predicate.is_some(), self.physical_predicate.is_some());

        let scan_sources = self.scan_sources.clone();
        let hive_partitions = self.hive_parts.clone();
        let hive_partitions_width = hive_partitions
            .as_deref()
            .map(|x| x[0].get_statistics().column_stats().len())
            .unwrap_or(0);
        let include_file_paths = self.file_options.include_file_paths.clone();
        let projected_arrow_schema = self.projected_arrow_schema.clone().unwrap();
        let row_index = self.file_options.row_index.clone();
        let physical_predicate = self.physical_predicate.clone();
        let ideal_morsel_size = get_ideal_morsel_size();
        let min_values_per_thread = self.config.min_values_per_thread;

        let mut use_prefiltered = physical_predicate.is_some()
            && matches!(
                self.options.parallel,
                ParallelStrategy::Auto | ParallelStrategy::Prefiltered
            );

        let predicate_arrow_field_indices = if use_prefiltered {
            let v = physical_predicate
                .as_ref()
                .unwrap()
                .live_variables()
                .and_then(|x| {
                    let mut out = x
                        .iter()
                        // Can be `None` - if the column is e.g. a hive column, or the row index column.
                        .filter_map(|x| projected_arrow_schema.index_of(x))
                        .collect::<Vec<_>>();

                    out.sort_unstable();
                    out.dedup();
                    // There is at least one non-predicate column, or pre-filtering was
                    // explicitly requested (only useful for testing).
                    (out.len() < projected_arrow_schema.len()
                        || matches!(self.options.parallel, ParallelStrategy::Prefiltered))
                    .then_some(out)
                });

            use_prefiltered &= v.is_some();

            v.unwrap_or_default()
        } else {
            vec![]
        };

        let use_prefiltered = use_prefiltered.then(PrefilterMaskSetting::init_from_env);

        let non_predicate_arrow_field_indices = if use_prefiltered.is_some() {
            filtered_range(
                predicate_arrow_field_indices.as_slice(),
                projected_arrow_schema.len(),
            )
        } else {
            vec![]
        };

        if use_prefiltered.is_some() && self.verbose {
            eprintln!(
                "[ParquetSource]: Pre-filtered decode enabled ({} live, {} non-live)",
                predicate_arrow_field_indices.len(),
                non_predicate_arrow_field_indices.len()
            )
        }

        let predicate_arrow_field_mask = if use_prefiltered.is_some() {
            let mut out = vec![false; projected_arrow_schema.len()];
            for i in predicate_arrow_field_indices.iter() {
                out[*i] = true;
            }
            out
        } else {
            vec![]
        };

        RowGroupDecoder {
            scan_sources,
            hive_partitions,
            hive_partitions_width,
            include_file_paths,
            projected_arrow_schema,
            row_index,
            physical_predicate,
            use_prefiltered,
            predicate_arrow_field_indices,
            non_predicate_arrow_field_indices,
            predicate_arrow_field_mask,
            ideal_morsel_size,
            min_values_per_thread,
        }
    }

    pub(super) fn init_projected_arrow_schema(&mut self) {
        let reader_schema = self
            .file_info
            .reader_schema
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap_left()
            .clone();

        self.projected_arrow_schema = Some(
            if let Some(columns) = self.file_options.with_columns.as_deref() {
                Arc::new(
                    columns
                        .iter()
                        .map(|x| {
                            let (_, k, v) = reader_schema.get_full(x).unwrap();
                            (k.clone(), v.clone())
                        })
                        .collect(),
                )
            } else {
                reader_schema.clone()
            },
        );

        if self.verbose {
            eprintln!(
                "[ParquetSource]: {} / {} parquet columns to be projected from {} files",
                self.projected_arrow_schema
                    .as_ref()
                    .map_or(reader_schema.len(), |x| x.len()),
                reader_schema.len(),
                self.scan_sources.len(),
            );
        }
    }
}

/// Returns 0..len in a Vec, excluding indices in `exclude`.
/// `exclude` needs to be a sorted list of unique values.
fn filtered_range(exclude: &[usize], len: usize) -> Vec<usize> {
    let mut j = 0;

    (0..len)
        .filter(|&i| {
            if j == exclude.len() || i != exclude[j] {
                true
            } else {
                j += 1;
                false
            }
        })
        .collect()
}

mod tests {

    #[test]
    fn test_filtered_range() {
        use super::filtered_range;
        assert_eq!(filtered_range(&[1, 3], 7).as_slice(), &[0, 2, 4, 5, 6]);
        assert_eq!(filtered_range(&[1, 6], 7).as_slice(), &[0, 2, 3, 4, 5]);
    }
}
