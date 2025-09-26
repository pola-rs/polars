use std::num::NonZeroUsize;

use polars_core::frame::DataFrame;
use polars_error::PolarsResult;
use polars_plan::prelude::PlanCallback;

use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::nodes::ComputeNode;
use crate::pipe::{RecvPort, SendPort};

pub struct CallbackSinkNode {
    function: PlanCallback<DataFrame, bool>,
    maintain_order: bool,

    buffer: DataFrame,
    chunk_size: Option<NonZeroUsize>,
    is_done: bool,
}

impl CallbackSinkNode {
    pub fn new(
        function: PlanCallback<DataFrame, bool>,
        maintain_order: bool,
        chunk_size: Option<NonZeroUsize>,
    ) -> Self {
        Self {
            function,
            maintain_order,

            buffer: DataFrame::empty(),
            chunk_size,
            is_done: false,
        }
    }
}

impl ComputeNode for CallbackSinkNode {
    fn name(&self) -> &str {
        "sink_batches"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.is_empty());

        if self.is_done || recv[0] == PortState::Done {
            recv[0] = PortState::Done;

            // Flush the last buffer
            if !self.buffer.is_empty() && !self.is_done {
                let function = self.function.clone();
                let df = std::mem::take(&mut self.buffer);

                assert!(
                    self.chunk_size
                        .is_some_and(|chunk_size| self.buffer.height() <= chunk_size.into())
                );
                state.spawn_subphase_task(async move {
                    polars_io::pl_async::get_runtime()
                        .spawn_blocking(move || function.call(df))
                        .await
                        .unwrap()?;
                    Ok(())
                });
                return Ok(());
            }
        } else {
            recv[0] = PortState::Ready;
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1 && send_ports.is_empty());
        let mut recv = recv_ports[0]
            .take()
            .unwrap()
            .serial_with_maintain_order(self.maintain_order);

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while !self.is_done
                && let Ok(m) = recv.recv().await
            {
                let (df, _, _, consume_token) = m.into_inner();

                // @NOTE: This also performs schema validation.
                self.buffer.vstack_mut(&df)?;

                while !self.buffer.is_empty()
                    && self
                        .chunk_size
                        .is_none_or(|chunk_size| self.buffer.height() >= chunk_size.into())
                {
                    let chunk_size = self.chunk_size.map_or(usize::MAX, Into::into);

                    let df;
                    (df, self.buffer) = self
                        .buffer
                        .split_at(self.buffer.height().min(chunk_size) as i64);

                    let function = self.function.clone();
                    let should_stop = polars_io::pl_async::get_runtime()
                        .spawn_blocking(move || function.call(df))
                        .await
                        .unwrap()?;

                    if should_stop {
                        self.is_done = true;
                        break;
                    }
                }
                drop(consume_token);
                // Increase the backpressure. Only free up a pipeline when the morsel has been
                // processed in its entirety.
            }

            Ok(())
        }));
    }
}
