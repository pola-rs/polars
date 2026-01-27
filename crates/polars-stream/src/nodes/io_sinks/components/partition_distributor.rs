use std::collections::VecDeque;
use std::sync::Arc;

use polars_core::prelude::{InitHashMaps, PlHashSet, PlIndexMap};
use polars_error::PolarsResult;
use polars_plan::dsl::file_provider::FileProviderArgs;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use crate::async_executor::{self, TaskPriority};
use crate::nodes::io_sinks::components::error_capture::{ErrorCapture, ErrorHandle};
use crate::nodes::io_sinks::components::file_sink::FileSinkPermit;
use crate::nodes::io_sinks::components::partition_key::PartitionKey;
use crate::nodes::io_sinks::components::partition_morsel_sender::PartitionMorselSender;
use crate::nodes::io_sinks::components::partition_sink_starter::PartitionSinkStarter;
use crate::nodes::io_sinks::components::partition_state::PartitionState;
use crate::nodes::io_sinks::components::partitioner::{self, PartitionedDataFrames};
use crate::nodes::io_sinks::components::size::RowCountAndSize;

pub struct PartitionDistributor {
    pub node_name: PlSmallStr,
    pub partitioned_dfs_rx: tokio::sync::mpsc::Receiver<
        async_executor::AbortOnDropHandle<PolarsResult<PartitionedDataFrames>>,
    >,
    pub partition_morsel_sender: PartitionMorselSender,
    pub error_capture: ErrorCapture,
    pub error_handle: ErrorHandle,
    pub max_open_sinks: usize,
    pub open_sinks_semaphore: Arc<tokio::sync::Semaphore>,
    pub partition_sink_starter: PartitionSinkStarter,
    pub no_partition_keys: bool,
    pub verbose: bool,
}

impl PartitionDistributor {
    pub async fn run(self) -> PolarsResult<()> {
        let PartitionDistributor {
            node_name,
            mut partitioned_dfs_rx,
            partition_morsel_sender,
            error_capture,
            error_handle,
            max_open_sinks,
            open_sinks_semaphore,
            partition_sink_starter,
            no_partition_keys,
            verbose,
        } = self;

        // No permits should have been acquired before this point.
        assert_eq!(open_sinks_semaphore.available_permits(), max_open_sinks);

        let mut partitions: PlIndexMap<PartitionKey, PartitionState> = Default::default();

        // Indices of partitions that have morsels ready to send.
        let mut ready_to_send_partitions: VecDeque<usize> = VecDeque::with_capacity(max_open_sinks);

        // Indices into `partitions` where a `FileSink` is currently open.
        let mut open_sinks: PlHashSet<usize> = PlHashSet::with_capacity(max_open_sinks);

        // How many `FileSink`s were forcibly closed to reclaim a sink permit.
        let mut forced_sink_closes: usize = 0;

        loop {
            if error_handle.has_errored() {
                return Err(error_handle.join().await.unwrap_err());
            }

            let Some(task_handle) = partitioned_dfs_rx.recv().await else {
                break;
            };

            let PartitionedDataFrames {
                partitions_vec,
                input_size,
                input_wait_token,
            } = task_handle.await?;

            for partitioner::Partition { key, keys_df, df } in partitions_vec {
                let partition_index = if let Some((index, ..)) = partitions.get_full(&key) {
                    index
                } else {
                    partitions
                        .insert_full(
                            key,
                            PartitionState {
                                buffered_rows: df.clear(),
                                keys_df: Arc::new(keys_df),
                                ..Default::default()
                            },
                        )
                        .0
                };

                let estimated_size: f64 = (input_size.num_bytes as f64)
                    * (df.height() as f64 / input_size.num_rows as f64);
                let estimated_size: u64 = estimated_size as _;

                let (_, partition_data) = partitions.get_index_mut(partition_index).unwrap();

                let num_rows = IdxSize::try_from(df.height()).unwrap();

                partition_data.buffered_rows.vstack_mut_owned_unchecked(df);
                partition_data.total_size = partition_data.total_size.add(RowCountAndSize {
                    num_rows,
                    num_bytes: estimated_size,
                })?;

                let buffered_size = partition_data.buffered_size();

                let num_ready_to_send_rows = partition_morsel_sender
                    .takeable_rows_provider
                    .num_rows_takeable_from(buffered_size, false);

                if num_ready_to_send_rows.is_some() {
                    if partition_data.file_sink_task_data.is_none()
                        && let Ok(file_permit) = open_sinks_semaphore.clone().try_acquire_owned()
                    {
                        partition_data.file_sink_task_data =
                            Some(partition_sink_starter.start_sink(
                                FileProviderArgs {
                                    index_in_partition: partition_data.num_sink_opens,
                                    partition_keys: partition_data.keys_df.clone(),
                                },
                                partition_data.sinked_size,
                                file_permit,
                            )?);
                        partition_data.num_sink_opens += 1;
                        open_sinks.insert(partition_index);
                    }

                    if partition_data.file_sink_task_data.is_some() {
                        ready_to_send_partitions.push_front(partition_index)
                    } else {
                        ready_to_send_partitions.push_back(partition_index);
                    }
                }
            }

            for partition_index in ready_to_send_partitions.drain(..) {
                let (_, partition) = partitions.get_index_mut(partition_index).unwrap();

                let partition: &mut PartitionState = if partition.file_sink_task_data.is_none() {
                    assert!(!open_sinks.contains(&partition_index));

                    let p: &mut PartitionState;

                    let file_permit: FileSinkPermit =
                        if let Ok(file_permit) = open_sinks_semaphore.clone().try_acquire_owned() {
                            p = partition;
                            file_permit
                        } else if open_sinks.len() < max_open_sinks {
                            p = partition;
                            // There are excess (>1) permits on a partition that are in the process of being closed.
                            open_sinks_semaphore.clone().acquire_owned().await.unwrap()
                        } else {
                            // Close a file sink and use the permit reclaimed from it.
                            let i = *open_sinks
                                .iter()
                                .min_by_key(|i| partitions.get_index(**i).unwrap().1.num_sink_opens)
                                .unwrap();

                            assert!(open_sinks.remove(&i));
                            forced_sink_closes += 1;

                            let task_data = partitions
                                .get_index_mut(i)
                                .unwrap()
                                .1
                                .file_sink_task_data
                                .take()
                                .unwrap();

                            p = partitions.get_index_mut(partition_index).unwrap().1;

                            task_data.close().await?
                        };

                    let partition = p;

                    let file_sink_task_data = partition_sink_starter.start_sink(
                        FileProviderArgs {
                            index_in_partition: partition.num_sink_opens,
                            partition_keys: partition.keys_df.clone(),
                        },
                        partition.sinked_size,
                        file_permit,
                    )?;

                    partition.num_sink_opens += 1;
                    partition.file_sink_task_data = Some(file_sink_task_data);
                    open_sinks.insert(partition_index);
                    partition
                } else {
                    partition
                };

                partition_morsel_sender
                    .send_morsels(partition, false)
                    .await?;
            }

            drop(input_wait_token);
        }

        if verbose {
            eprintln!("{node_name}: Begin finalize");
        }

        assert!(ready_to_send_partitions.is_empty());

        if no_partition_keys {
            assert_eq!(partitions.len(), 1);
            let partition = partitions.get_index(0).unwrap().1;
            assert_eq!(partition.keys_df.width(), 0);
        }

        // Statistics
        let num_partitions = partitions.len();
        let mut finalize_flush_size = RowCountAndSize::default();
        let mut total_size = RowCountAndSize::default();
        let mut total_sink_opens: usize = 0;

        // Finalize partitions with existing open sinks first.
        let indices_iter = open_sinks
            .iter()
            .copied()
            .chain((0..partitions.len()).filter(|i| !open_sinks.contains(i)));

        for partition_index in indices_iter {
            if error_handle.has_errored() {
                return Err(error_handle.join().await.unwrap_err());
            }

            let partition: &mut PartitionState =
                partitions.get_index_mut(partition_index).unwrap().1;

            let residual_size = partition.buffered_size();
            finalize_flush_size = finalize_flush_size.saturating_add(residual_size);

            assert_eq!(
                usize::try_from(residual_size.num_rows).unwrap(),
                partition.buffered_rows.height()
            );

            if (residual_size.num_rows > 0 || (no_partition_keys && partition.num_sink_opens == 0))
                && partition.file_sink_task_data.is_none()
            {
                // No need for sink-closing logic here. All permits during finalize are guaranteed
                // to be dropped, and traversal starts with open sinks first.
                let file_permit = open_sinks_semaphore.clone().acquire_owned().await.unwrap();

                let file_sink_task_data = partition_sink_starter.start_sink(
                    FileProviderArgs {
                        index_in_partition: partition.num_sink_opens,
                        partition_keys: partition.keys_df.clone(),
                    },
                    partition.sinked_size,
                    file_permit,
                )?;

                partition.num_sink_opens += 1;
                partition.file_sink_task_data = Some(file_sink_task_data);
            }

            if residual_size.num_rows > 0 {
                partition_morsel_sender
                    .send_morsels(partition, true)
                    .await?;

                assert_eq!(
                    partition.sinked_size.num_rows,
                    partition.total_size.num_rows
                );
            }

            total_size = total_size.saturating_add(partition.total_size);
            total_sink_opens = total_sink_opens.saturating_add(partition.num_sink_opens);

            if let Some(file_sink_task_data) = partition.file_sink_task_data.take() {
                async_executor::spawn(
                    TaskPriority::Low,
                    error_capture
                        .clone()
                        .wrap_future(file_sink_task_data.close()),
                );
            }
        }

        drop(error_capture);
        drop(partition_morsel_sender);

        if verbose {
            eprintln!("{node_name}: PartitionDistributor: Join tasks")
        }

        error_handle.join().await?;

        if verbose {
            eprintln!(
                "\
                {node_name}: Statistics: \
                num_partitions: {}, \
                total_size: {:?}, \
                finalize_flush_size: {:?} ({:.3}% total rows, {:.3}% total bytes), \
                total_sink_opens: {}, \
                forced_sink_closes: {} ({:.3}% total, {:.3}% max)",
                num_partitions,
                total_size,
                finalize_flush_size,
                100f64 * (finalize_flush_size.num_rows as f64 / total_size.num_rows.max(1) as f64),
                100f64
                    * (finalize_flush_size.num_bytes as f64 / total_size.num_bytes.max(1) as f64),
                total_sink_opens,
                forced_sink_closes,
                100f64 * (forced_sink_closes as f64 / total_sink_opens.max(1) as f64),
                100f64
                    * (forced_sink_closes as f64
                        / total_sink_opens.saturating_sub(num_partitions).max(1) as f64)
                        .min(1.0)
            );
        }

        Ok(())
    }
}
