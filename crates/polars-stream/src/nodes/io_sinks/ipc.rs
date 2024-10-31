use std::fs::{File, OpenOptions};
use std::path::Path;

use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_io::ipc::{BatchedWriter, IpcWriter, IpcWriterOptions};
use polars_io::SerWriter;

use crate::nodes::{ComputeNode, JoinHandle, PortState, TaskPriority, TaskScope};
use crate::pipe::{RecvPort, SendPort};

pub struct IpcSinkNode {
    is_finished: bool,
    writer: BatchedWriter<File>,
}

impl IpcSinkNode {
    pub fn new(
        input_schema: SchemaRef,
        path: &Path,
        write_options: &IpcWriterOptions,
    ) -> PolarsResult<Self> {
        let file = OpenOptions::new().write(true).open(path)?;
        let writer = IpcWriter::new(file)
            .with_compression(write_options.compression)
            .batched(&input_schema)?;

        Ok(Self {
            is_finished: false,
            writer,
        })
    }
}

impl ComputeNode for IpcSinkNode {
    fn name(&self) -> &str {
        "ipc_sink"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(send.is_empty());
        assert!(recv.len() == 1);

        if recv[0] == PortState::Done && !self.is_finished {
            // @NOTE: This function can be called afterwards multiple times. So make sure to only
            // finish the writer once.
            self.is_finished = true;
            self.writer.finish()?;
        }

        // We are always ready to receive, unless the sender is done, then we're
        // also done.
        if recv[0] != PortState::Done {
            recv[0] = PortState::Ready;
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
        assert!(send_ports.is_empty());
        assert!(recv_ports.len() == 1);
        let mut receiver = recv_ports[0].take().unwrap().serial();

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = receiver.recv().await {
                self.writer.write_batch(&morsel.into_df())?;
            }

            Ok(())
        }));
    }
}
