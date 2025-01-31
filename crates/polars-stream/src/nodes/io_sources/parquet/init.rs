use std::future::Future;
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_error::PolarsResult;
use polars_io::prelude::ParallelStrategy;
use polars_io::prelude::_internal::PrefilterMaskSetting;

use super::row_group_data_fetch::RowGroupDataFetcher;
use super::row_group_decode::RowGroupDecoder;
use super::{AsyncTaskData, ParquetSourceNode};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::morsel::get_ideal_morsel_size;
use crate::nodes::{MorselSeq, TaskPriority};
use crate::utils::task_handles_ext::{self, AbortOnDropHandle};
use crate::{async_executor, DEFAULT_DISTRIBUTOR_BUFFER_SIZE};

impl ParquetSourceNode {
    /// # Panics
    /// Panics if called more than once.
    async fn shutdown_impl(
        async_task_data: Arc<tokio::sync::Mutex<Option<AsyncTaskData>>>,
        verbose: bool,
    ) -> PolarsResult<()> {
        if verbose {
            eprintln!("[ParquetSource]: Shutting down");
        }

        let (raw_morsel_receivers, morsel_stream_task_handle) =
            async_task_data.try_lock().unwrap().take().unwrap();

        drop(raw_morsel_receivers);
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
    pub(super) fn init_raw_morsel_distributor(&mut self) -> AsyncTaskData {
        let verbose = self.verbose;
        let io_runtime = polars_io::pl_async::get_runtime();

        let use_statistics = self.options.use_statistics;

        let (mut raw_morsel_sender, raw_morsel_receivers) =
            distributor_channel(self.config.num_pipelines, DEFAULT_DISTRIBUTOR_BUFFER_SIZE);
        if let Some((_, 0)) = self.file_options.slice {
            return (
                raw_morsel_receivers,
                task_handles_ext::AbortOnDropHandle(io_runtime.spawn(std::future::ready(Ok(())))),
            );
        }

        let reader_schema = self.schema.clone().unwrap();

        let (normalized_slice_oneshot_rx, metadata_rx, metadata_task) =
            self.init_metadata_fetcher();

        let row_group_prefetch_size = self.config.row_group_prefetch_size;
        let projection = self.file_options.with_columns.clone();
        let predicate = self.predicate.clone();
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

        let ideal_morsel_size = get_ideal_morsel_size();

        if verbose {
            eprintln!("[ParquetSource]: ideal_morsel_size: {}", ideal_morsel_size);
        }

        // Prefetch loop (spawns prefetches on the tokio scheduler).
        let (prefetch_send, mut prefetch_recv) =
            tokio::sync::mpsc::channel(row_group_prefetch_size);
        let prefetch_task = AbortOnDropHandle(io_runtime.spawn(async move {
            let slice_range = {
                let Ok(slice) = normalized_slice_oneshot_rx.await else {
                    // If we are here then the producer probably errored.
                    drop(row_group_data_fetcher);
                    return PolarsResult::Ok(());
                };

                slice.map(|(offset, len)| offset..offset + len)
            };

            row_group_data_fetcher.slice_range = slice_range;

            loop {
                let Some(prefetch) = row_group_data_fetcher.next().await else {
                    break;
                };
                if prefetch_send.send(prefetch?).await.is_err() {
                    break;
                }
            }
            PolarsResult::Ok(())
        }));

        // Decode loop (spawns decodes on the computational executor).
        let (decode_send, mut decode_recv) = tokio::sync::mpsc::channel(self.config.num_pipelines);
        let decode_task = AbortOnDropHandle(io_runtime.spawn(async move {
            while let Some(prefetch) = prefetch_recv.recv().await {
                let row_group_data = prefetch.await.unwrap()?;
                let row_group_decoder = row_group_decoder.clone();
                let decode_fut = async_executor::spawn(TaskPriority::High, async move {
                    row_group_decoder.row_group_data_to_df(row_group_data).await
                });
                if decode_send.send(decode_fut).await.is_err() {
                    break;
                }
            }
            PolarsResult::Ok(())
        }));

        // Distributes morsels across pipelines. This does not perform any CPU or I/O bound work -
        // it is purely a dispatch loop. Run on the computational executor to reduce context switches.
        let last_morsel_min_split = self.config.num_pipelines;
        let distribute_task = async_executor::spawn(TaskPriority::High, async move {
            let mut morsel_seq = MorselSeq::default();

            // Decode first non-empty morsel.
            let mut next = None;
            loop {
                let Some(decode_fut) = decode_recv.recv().await else {
                    break;
                };
                let df = decode_fut.await?;
                if df.is_empty() {
                    continue;
                }
                next = Some(df);
                break;
            }

            while let Some(df) = next.take() {
                // Try to decode the next non-empty morsel first, so we know
                // whether the df is the last morsel.
                loop {
                    let Some(decode_fut) = decode_recv.recv().await else {
                        break;
                    };
                    let next_df = decode_fut.await?;
                    if next_df.is_empty() {
                        continue;
                    }
                    next = Some(next_df);
                    break;
                }

                for df in split_to_morsels(
                    &df,
                    ideal_morsel_size,
                    next.is_none(),
                    last_morsel_min_split,
                ) {
                    if raw_morsel_sender.send((df, morsel_seq)).await.is_err() {
                        return Ok(());
                    }
                    morsel_seq = morsel_seq.successor();
                }
            }
            PolarsResult::Ok(())
        });

        let join_task = io_runtime.spawn(async move {
            metadata_task.await.unwrap()?;
            prefetch_task.await.unwrap()?;
            decode_task.await.unwrap()?;
            distribute_task.await?;
            Ok(())
        });

        (raw_morsel_receivers, AbortOnDropHandle(join_task))
    }

    /// Creates a `RowGroupDecoder` that turns `RowGroupData` into DataFrames.
    /// This must be called AFTER the following have been initialized:
    /// * `self.projected_arrow_schema`
    /// * `self.physical_predicate`
    pub(super) fn init_row_group_decoder(&self) -> RowGroupDecoder {
        let scan_sources = self.scan_sources.clone();
        let hive_partitions = self.hive_parts.clone();
        let hive_partitions_width = hive_partitions
            .as_deref()
            .map(|x| x[0].get_statistics().column_stats().len())
            .unwrap_or(0);
        let include_file_paths = self.file_options.include_file_paths.clone();
        let projected_arrow_schema = self.projected_arrow_schema.clone().unwrap();
        let row_index = self.row_index.clone();
        let min_values_per_thread = self.config.min_values_per_thread;

        let mut use_prefiltered = self.predicate.is_some()
            && matches!(
                self.options.parallel,
                ParallelStrategy::Auto | ParallelStrategy::Prefiltered
            );

        let predicate_arrow_field_indices = if use_prefiltered {
            let predicate = self.predicate.as_ref().unwrap();
            let v = (!predicate.live_columns.is_empty())
                .then(|| {
                    let mut out = predicate
                        .live_columns
                        .iter()
                        // Can be `None` - if the column is e.g. a hive column, or the row index column.
                        .filter_map(|x| projected_arrow_schema.index_of(x))
                        .collect::<Vec<_>>();

                    out.sort_unstable();

                    // There is at least one non-predicate column, or pre-filtering was
                    // explicitly requested (only useful for testing).
                    (out.len() < projected_arrow_schema.len()
                        || matches!(self.options.parallel, ParallelStrategy::Prefiltered))
                    .then_some(out)
                })
                .flatten();

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

        RowGroupDecoder {
            scan_sources,
            hive_partitions,
            hive_partitions_width,
            include_file_paths,
            reader_schema: self.schema.clone().unwrap(),
            projected_arrow_schema,
            row_index,
            predicate: self.predicate.clone(),
            use_prefiltered,
            predicate_arrow_field_indices,
            non_predicate_arrow_field_indices,
            min_values_per_thread,
        }
    }

    pub(super) fn init_projected_arrow_schema(&mut self) {
        let reader_schema = self.schema.clone().unwrap();

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
    if cfg!(debug_assertions) {
        assert!(exclude.windows(2).all(|x| x[1] > x[0]));
    }

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

fn split_to_morsels(
    df: &DataFrame,
    ideal_morsel_size: usize,
    last_morsel: bool,
    last_morsel_min_split: usize,
) -> impl Iterator<Item = DataFrame> + '_ {
    let mut n_morsels = if df.height() > 3 * ideal_morsel_size / 2 {
        // num_rows > (1.5 * ideal_morsel_size)
        (df.height() / ideal_morsel_size).max(2)
    } else {
        1
    };

    if last_morsel {
        n_morsels = n_morsels.max(last_morsel_min_split);
    }

    let rows_per_morsel = df.height().div_ceil(n_morsels).max(1);

    (0..i64::try_from(df.height()).unwrap())
        .step_by(rows_per_morsel)
        .map(move |offset| df.slice(offset, rows_per_morsel))
        .filter(|df| !df.is_empty())
}

mod tests {

    #[test]
    fn test_filtered_range() {
        use super::filtered_range;
        assert_eq!(filtered_range(&[1, 3], 7).as_slice(), &[0, 2, 4, 5, 6]);
        assert_eq!(filtered_range(&[1, 6], 7).as_slice(), &[0, 2, 3, 4, 5]);
    }
}
