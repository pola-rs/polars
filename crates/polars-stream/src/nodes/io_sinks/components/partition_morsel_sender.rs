use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_error::PolarsResult;
use polars_plan::dsl::file_provider::FileProviderArgs;
use polars_utils::IdxSize;

use crate::async_executor::{self, TaskPriority};
use crate::nodes::io_sinks::components::error_capture::ErrorCapture;
use crate::nodes::io_sinks::components::file_sink::FileSinkPermit;
use crate::nodes::io_sinks::components::hstack_columns::HStackColumns;
use crate::nodes::io_sinks::components::partition_sink_starter::PartitionSinkStarter;
use crate::nodes::io_sinks::components::partition_state::PartitionState;
use crate::nodes::io_sinks::components::sink_morsel::SinkMorsel;
use crate::nodes::io_sinks::components::size::{
    NonZeroRowCountAndSize, RowCountAndSize, TakeableRowsProvider,
};

pub struct PartitionMorselSender {
    /// Note: Must be <= `file_size_limit` if there is one.
    pub takeable_rows_provider: TakeableRowsProvider,
    pub file_size_limit: NonZeroRowCountAndSize,
    pub inflight_morsel_semaphore: Arc<tokio::sync::Semaphore>,
    pub open_sinks_semaphore: Arc<tokio::sync::Semaphore>,
    pub partition_sink_starter: PartitionSinkStarter,
    /// For include_key: true
    pub hstack_keys: Option<HStackColumns>,
    pub error_capture: ErrorCapture,
}

impl PartitionMorselSender {
    /// # Panics
    /// Panics if `partition.file_sink_task_data` is `None`.
    pub async fn send_morsels(
        &self,
        partition: &mut PartitionState,
        flush: bool,
    ) -> PolarsResult<()> {
        loop {
            let file_sink_task_data = partition.file_sink_task_data.as_mut().unwrap();

            let mut used_row_capacity: RowCountAndSize;
            let mut available_row_capacity: RowCountAndSize;

            macro_rules! calc_used_and_available_capacities {
                ($file_sink_task_data:expr) => {{
                    used_row_capacity = partition
                        .sinked_size
                        .checked_sub($file_sink_task_data.start_position)
                        .unwrap();
                    available_row_capacity = if self.file_size_limit.get() == RowCountAndSize::MAX {
                        RowCountAndSize::MAX
                    } else {
                        let file_size_limit = self.file_size_limit.get();
                        RowCountAndSize {
                            num_rows: IdxSize::checked_sub(
                                file_size_limit.num_rows,
                                used_row_capacity.num_rows,
                            )
                            .unwrap(),
                            num_bytes: u64::saturating_sub(
                                file_size_limit.num_bytes,
                                used_row_capacity.num_bytes,
                            ),
                        }
                    };
                }};
            }

            calc_used_and_available_capacities!(file_sink_task_data);

            if used_row_capacity.num_rows == 0 && available_row_capacity.num_rows == 0 {
                available_row_capacity = RowCountAndSize {
                    num_rows: 1,
                    num_bytes: u64::MAX,
                };
            }

            let buffered_size = partition.buffered_size();

            if buffered_size.num_rows == 0 {
                return Ok(());
            }

            let Some(num_rows_to_take) = self
                .takeable_rows_provider
                .num_rows_takeable_from(buffered_size, flush)
            else {
                return Ok(());
            };

            let file_min_available_rows_for_byte_size =
                NonZeroRowCountAndSize::get(self.takeable_rows_provider.max_size)
                    .num_rows
                    .saturating_sub(used_row_capacity.num_rows);

            let max_takeable_rows: IdxSize = available_row_capacity
                .num_rows_takeable_from(buffered_size, file_min_available_rows_for_byte_size);

            let start_new_sink = max_takeable_rows == 0;
            let num_rows_to_take = if start_new_sink {
                num_rows_to_take
            } else {
                num_rows_to_take.min(max_takeable_rows)
            };

            if start_new_sink {
                assert!(used_row_capacity.num_rows > 0);
                let handle = partition.file_sink_task_data.take().unwrap().close();

                let file_permit: FileSinkPermit =
                    if let Ok(permit) = self.open_sinks_semaphore.clone().try_acquire_owned() {
                        async_executor::spawn(
                            TaskPriority::Low,
                            self.error_capture.clone().wrap_future(handle),
                        );

                        permit
                    } else {
                        handle.await?
                    };

                partition.file_sink_task_data = Some(self.partition_sink_starter.start_sink(
                    FileProviderArgs {
                        index_in_partition: partition.num_sink_opens,
                        partition_keys: partition.keys_df.clone(),
                    },
                    partition.sinked_size,
                    file_permit,
                )?);
                partition.num_sink_opens += 1;
            }

            let file_sink_task_data = partition.file_sink_task_data.as_mut().unwrap();

            if start_new_sink {
                calc_used_and_available_capacities!(file_sink_task_data);
            }

            let (df, remaining) = partition.buffered_rows.split_at(
                #[allow(clippy::unnecessary_fallible_conversions)]
                i64::try_from(num_rows_to_take).unwrap(),
            );

            partition.buffered_rows = remaining;

            let morsel_permit = self
                .inflight_morsel_semaphore
                .clone()
                .acquire_owned()
                .await
                .unwrap();

            let mut morsel = SinkMorsel::new(df, morsel_permit);

            let morsel_height: IdxSize = IdxSize::try_from(morsel.df().height()).unwrap();

            debug_assert!(
                self.takeable_rows_provider.max_size.get().num_rows
                    <= self.file_size_limit.get().num_rows
            );
            debug_assert!(morsel_height <= self.takeable_rows_provider.max_size.get().num_rows);

            assert!((1..=available_row_capacity.num_rows).contains(&morsel_height));

            if let Some(hstack_keys) = self.hstack_keys.as_ref() {
                let columns = morsel.df().columns();
                let height = morsel.df().height();
                let new_columns = hstack_keys.hstack_columns_broadcast(
                    height,
                    columns,
                    partition.keys_df.columns(),
                );

                *morsel.df_mut() = unsafe { DataFrame::new_unchecked(height, new_columns) };
            };

            if file_sink_task_data.morsel_tx.send(morsel).await.is_err() {
                let handle = partition.file_sink_task_data.take().unwrap().close();
                return Err(handle.await.unwrap_err());
            }

            partition.sinked_size = partition
                .sinked_size
                .add_delta(morsel_height, partition.total_size)
                .unwrap();
        }
    }
}
