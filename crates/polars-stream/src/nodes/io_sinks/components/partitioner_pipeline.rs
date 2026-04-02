use std::sync::Arc;

use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;

use crate::async_executor::{self, TaskPriority};
use crate::async_primitives::connector;
use crate::morsel::Morsel;
use crate::nodes::io_sinks::components::partitioner::{PartitionedDataFrames, Partitioner};

pub struct PartitionerPipeline {
    pub morsel_rx: connector::Receiver<Morsel>,
    pub partitioner: Arc<Partitioner>,
    pub inflight_morsel_semaphore: Arc<tokio::sync::Semaphore>,
    pub partitioned_dfs_tx: tokio::sync::mpsc::Sender<
        async_executor::AbortOnDropHandle<PolarsResult<PartitionedDataFrames>>,
    >,
    pub in_memory_exec_state: Arc<ExecutionState>,
}

impl PartitionerPipeline {
    pub async fn run(self) {
        let PartitionerPipeline {
            mut morsel_rx,
            partitioner,
            inflight_morsel_semaphore,
            partitioned_dfs_tx,
            in_memory_exec_state,
        } = self;

        loop {
            // Acquire a single permit to have backpressure. This is not attached to the send as the
            // morsels from here do not count towards the in-flight morsel limit.
            let permit = inflight_morsel_semaphore.acquire().await.unwrap();
            drop(permit);

            let Ok(morsel) = morsel_rx.recv().await else {
                return;
            };

            let partitioner = Arc::clone(&partitioner);
            let in_memory_exec_state = Arc::clone(&in_memory_exec_state);

            if partitioned_dfs_tx
                .send(async_executor::AbortOnDropHandle::new(
                    async_executor::spawn(TaskPriority::Low, async move {
                        partitioner
                            .partition_morsel(morsel, in_memory_exec_state.as_ref())
                            .await
                    }),
                ))
                .await
                .is_err()
            {
                return;
            }
        }
    }
}
