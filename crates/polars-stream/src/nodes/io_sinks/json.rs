use std::cmp::Reverse;

use polars_error::PolarsResult;
use polars_io::cloud::CloudOptions;
use polars_io::json::BatchedWriter;
use polars_plan::dsl::{SinkOptions, SinkTarget};
use polars_utils::priority::Priority;

use super::{SinkInputPort, SinkNode};
use crate::async_executor::spawn;
use crate::async_primitives::connector::Receiver;
use crate::execute::StreamingExecutionState;
use crate::nodes::io_sinks::parallelize_receive_task;
use crate::nodes::io_sinks::phase::PhaseOutcome;
use crate::nodes::{JoinHandle, TaskPriority};

pub struct NDJsonSinkNode {
    target: SinkTarget,
    sink_options: SinkOptions,
    cloud_options: Option<CloudOptions>,
}
impl NDJsonSinkNode {
    pub fn new(
        target: SinkTarget,
        sink_options: SinkOptions,
        cloud_options: Option<CloudOptions>,
    ) -> Self {
        Self {
            target,
            sink_options,
            cloud_options,
        }
    }
}

impl SinkNode for NDJsonSinkNode {
    fn name(&self) -> &str {
        "ndjson-sink"
    }

    fn is_sink_input_parallel(&self) -> bool {
        true
    }
    fn do_maintain_order(&self) -> bool {
        self.sink_options.maintain_order
    }

    fn spawn_sink(
        &mut self,
        recv_port_rx: Receiver<(PhaseOutcome, SinkInputPort)>,
        state: &StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let (pass_rxs, mut io_rx) = parallelize_receive_task(
            join_handles,
            recv_port_rx,
            state.num_pipelines,
            self.sink_options.maintain_order,
        );

        // 16MB
        const DEFAULT_ALLOCATION_SIZE: usize = 1 << 24;

        // Encode task.
        //
        // Task encodes the columns into their corresponding JSON encoding.
        join_handles.extend(pass_rxs.into_iter().map(|mut pass_rx| {
            spawn(TaskPriority::High, async move {
                // Amortize the allocations over time. If we see that we need to do way larger
                // allocations, we adjust to that over time.
                let mut allocation_size = DEFAULT_ALLOCATION_SIZE;

                while let Ok((mut rx, mut lin_tx)) = pass_rx.recv().await {
                    while let Ok(morsel) = rx.recv().await {
                        let (df, seq, _, consume_token) = morsel.into_inner();

                        let mut buffer = Vec::with_capacity(allocation_size);
                        let mut writer = BatchedWriter::new(&mut buffer);

                        writer.write_batch(&df)?;

                        allocation_size = allocation_size.max(buffer.len());
                        if lin_tx.insert(Priority(Reverse(seq), buffer)).await.is_err() {
                            return Ok(());
                        }
                        drop(consume_token); // Keep the consume_token until here to increase the
                        // backpressure.
                    }
                }

                PolarsResult::Ok(())
            })
        }));
        let cloud_options = self.cloud_options.clone();

        // IO task.
        //
        // Task that will actually do write to the target file.
        let sink_options = self.sink_options.clone();
        let target = self.target.clone();
        let io_task = polars_io::pl_async::get_runtime().spawn(async move {
            use tokio::io::AsyncWriteExt;

            let mut file = target
                .open_into_writeable_async(&sink_options, cloud_options.as_ref())
                .await?
                .try_into_async_writeable()?;

            while let Ok(mut lin_rx) = io_rx.recv().await {
                while let Some(Priority(_, buffer)) = lin_rx.get().await {
                    file.write_all(&buffer).await?;
                }
            }

            file.sync_on_close(sink_options.sync_on_close).await?;
            file.close().await?;

            PolarsResult::Ok(())
        });
        join_handles.push(spawn(TaskPriority::Low, async move {
            io_task
                .await
                .unwrap_or_else(|e| Err(std::io::Error::from(e).into()))
        }));
    }
}
