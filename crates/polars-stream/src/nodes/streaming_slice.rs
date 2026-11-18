use polars_utils::slice_enum::Slice;

use super::compute_node_prelude::*;

/// A node that passes through the rows of a positive slice.
/// Since the offset is non-negative this can be done in a streaming manner.
pub struct StreamingSliceNode {
    /// Relative to the rows not yet received.
    offset: usize,
    /// `None` runs to the end.
    length: Option<usize>,
}

impl StreamingSliceNode {
    pub fn new(offset: usize, length: Option<usize>) -> Self {
        Self { offset, length }
    }

    fn is_exhausted(&self) -> bool {
        self.length == Some(0)
    }
}

impl ComputeNode for StreamingSliceNode {
    fn name(&self) -> &str {
        "streaming-slice"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        if self.is_exhausted() {
            recv[0] = PortState::Done;
            send[0] = PortState::Done;
        } else {
            recv.swap_with_slice(send);
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
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);
        let mut recv = recv_ports[0].take().unwrap().serial();
        let mut send = send_ports[0].take().unwrap().serial();
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = recv.recv().await {
                let morsel = morsel
                    .map(|df| {
                        let height = df.height();
                        let range = Slice::new_positive(self.offset, self.length)
                            .map_or(0..height, |s| s.resolve(height));
                        self.offset = self.offset.saturating_sub(height);
                        self.length = self.length.map(|len| len - range.len());

                        if range.len() != height {
                            df.slice(range)
                        } else {
                            df
                        }
                    })
                    .await;

                // Technically not necessary, but it's nice to already tell the
                // source to stop producing more morsels as we won't be
                // interested in the results anyway.
                if self.is_exhausted() {
                    morsel.source_token().stop();
                }

                if morsel.height() > 0 && send.send(morsel).await.is_err() {
                    break;
                }

                if self.is_exhausted() {
                    break;
                }
            }

            Ok(())
        }))
    }
}
