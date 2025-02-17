use std::cmp::Reverse;
use std::path::{Path, PathBuf};

use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_io::json::BatchedWriter;
use polars_utils::priority::Priority;

use crate::async_primitives::linearizer::{Inserter, Linearizer};
use crate::nodes::{ComputeNode, JoinHandle, MorselSeq, PortState, TaskPriority, TaskScope};
use crate::pipe::{RecvPort, SendPort};
use crate::DEFAULT_LINEARIZER_BUFFER_SIZE;

type Linearized = Priority<Reverse<MorselSeq>, Vec<u8>>;
pub struct NDJsonSinkNode {
    path: PathBuf,

    // IO task related:
    senders: Vec<Inserter<Linearized>>,
    io_task: Option<tokio::task::JoinHandle<PolarsResult<()>>>,
}

impl NDJsonSinkNode {
    pub fn new(path: &Path) -> PolarsResult<Self> {
        Ok(Self {
            path: path.to_path_buf(),

            senders: Vec::new(),
            io_task: None,
        })
    }
}

impl ComputeNode for NDJsonSinkNode {
    fn name(&self) -> &str {
        "ndjson_sink"
    }

    fn initialize(&mut self, num_pipelines: usize) {
        // Encode tasks -> IO task
        let (mut linearizer, senders) =
            Linearizer::<Linearized>::new(num_pipelines, DEFAULT_LINEARIZER_BUFFER_SIZE);

        self.senders = senders;

        // IO task.
        //
        // Task that will actually do write to the target file.
        let io_runtime = polars_io::pl_async::get_runtime();

        let path = self.path.clone();
        self.io_task = Some(io_runtime.spawn(async move {
            use tokio::fs::OpenOptions;
            use tokio::io::AsyncWriteExt;

            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path.as_path())
                .await
                .map_err(|err| polars_utils::_limit_path_len_io_err(path.as_path(), err))?;

            while let Some(Priority(_, buffer)) = linearizer.get().await {
                file.write_all(&buffer).await?;
            }

            PolarsResult::Ok(())
        }));
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(send.is_empty());
        assert!(recv.len() == 1);

        // We are always ready to receive, unless the sender is done, then we're
        // also done.
        if recv[0] != PortState::Done {
            recv[0] = PortState::Ready;
        } else if let Some(io_task) = self.io_task.take() {
            // Stop the IO task from waiting for more morsels.
            self.senders.clear();

            polars_io::pl_async::get_runtime()
                .block_on(io_task)
                .unwrap_or_else(|e| Err(std::io::Error::from(e).into()))?;
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1);
        assert!(send_ports.is_empty());

        // .. -> Encode task
        let receivers = recv_ports[0].take().unwrap().parallel();

        // 16MB
        const DEFAULT_ALLOCATION_SIZE: usize = 1 << 24;

        // Encode task.
        //
        // Task encodes the columns into their corresponding JSON encoding.
        for (mut receiver, sender) in receivers.into_iter().zip(self.senders.iter_mut()) {
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                // Amortize the allocations over time. If we see that we need to do way larger
                // allocations, we adjust to that over time.
                let mut allocation_size = DEFAULT_ALLOCATION_SIZE;

                while let Ok(morsel) = receiver.recv().await {
                    let (df, seq, _, _) = morsel.into_inner();

                    let mut buffer = Vec::with_capacity(allocation_size);
                    let mut writer = BatchedWriter::new(&mut buffer);

                    writer.write_batch(&df)?;

                    allocation_size = allocation_size.max(buffer.len());
                    sender.insert(Priority(Reverse(seq), buffer)).await.unwrap();
                }

                PolarsResult::Ok(())
            }));
        }
    }
}
