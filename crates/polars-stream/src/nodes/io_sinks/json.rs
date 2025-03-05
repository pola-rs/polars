use std::cmp::Reverse;
use std::path::PathBuf;

use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_io::json::BatchedWriter;
use polars_plan::dsl::SinkOptions;
use polars_utils::priority::Priority;

use super::{SinkInputPort, SinkNode, SinkRecvPort};
use crate::async_executor::spawn;
use crate::async_primitives::linearizer::Linearizer;
use crate::nodes::io_sinks::{tokio_sync_on_close, DEFAULT_SINK_LINEARIZER_BUFFER_SIZE};
use crate::nodes::{JoinHandle, MorselSeq, TaskPriority};

type Linearized = Priority<Reverse<MorselSeq>, Vec<u8>>;
pub struct NDJsonSinkNode {
    path: PathBuf,
    sink_options: SinkOptions,
}
impl NDJsonSinkNode {
    pub fn new(path: PathBuf, sink_options: SinkOptions) -> Self {
        Self { path, sink_options }
    }
}

impl SinkNode for NDJsonSinkNode {
    fn name(&self) -> &str {
        "ndjson_sink"
    }

    fn is_sink_input_parallel(&self) -> bool {
        true
    }
    fn do_maintain_order(&self) -> bool {
        self.sink_options.maintain_order
    }

    fn spawn_sink(
        &mut self,
        num_pipelines: usize,
        recv_ports_recv: SinkRecvPort,
        _state: &ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let rxs = recv_ports_recv.parallel(join_handles);
        self.spawn_sink_once(
            num_pipelines,
            SinkInputPort::Parallel(rxs),
            _state,
            join_handles,
        );
    }

    fn spawn_sink_once(
        &mut self,
        num_pipelines: usize,
        recv_port: SinkInputPort,
        _state: &ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        // .. -> Encode task
        let rxs = recv_port.parallel();
        // Encode tasks -> IO task
        let (mut lin_rx, lin_txs) = Linearizer::<Linearized>::new_with_maintain_order(
            num_pipelines,
            DEFAULT_SINK_LINEARIZER_BUFFER_SIZE,
            self.sink_options.maintain_order,
        );

        // 16MB
        const DEFAULT_ALLOCATION_SIZE: usize = 1 << 24;

        // Encode task.
        //
        // Task encodes the columns into their corresponding JSON encoding.
        join_handles.extend(rxs.into_iter().zip(lin_txs).map(|(mut rx, mut lin_tx)| {
            spawn(TaskPriority::High, async move {
                // Amortize the allocations over time. If we see that we need to do way larger
                // allocations, we adjust to that over time.
                let mut allocation_size = DEFAULT_ALLOCATION_SIZE;

                while let Ok(morsel) = rx.recv().await {
                    let (df, seq, _, consume_token) = morsel.into_inner();

                    let mut buffer = Vec::with_capacity(allocation_size);
                    let mut writer = BatchedWriter::new(&mut buffer);

                    writer.write_batch(&df)?;

                    allocation_size = allocation_size.max(buffer.len());

                    // Must drop before linearizer insert or will deadlock.
                    drop(consume_token); // Keep the consume_token until here to increase the
                                         // backpressure.

                    if lin_tx.insert(Priority(Reverse(seq), buffer)).await.is_err() {
                        return Ok(());
                    }
                }

                PolarsResult::Ok(())
            })
        }));

        // IO task.
        //
        // Task that will actually do write to the target file.
        let sink_options = self.sink_options.clone();
        let path = self.path.clone();
        let io_task = polars_io::pl_async::get_runtime().spawn(async move {
            use tokio::fs::OpenOptions;
            use tokio::io::AsyncWriteExt;

            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path.as_path())
                .await
                .map_err(|err| polars_utils::_limit_path_len_io_err(path.as_path(), err))?;

            while let Some(Priority(_, buffer)) = lin_rx.get().await {
                file.write_all(&buffer).await?;
            }

            tokio_sync_on_close(sink_options.sync_on_close, &mut file).await?;
            PolarsResult::Ok(())
        });
        join_handles.push(spawn(TaskPriority::Low, async move {
            io_task
                .await
                .unwrap_or_else(|e| Err(std::io::Error::from(e).into()))
        }));
    }
}
