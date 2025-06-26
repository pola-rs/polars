use super::compute_node_prelude::*;

/// A node that will pass-through up to length rows, starting at start_offset.
/// Since start_offset must be non-negative this can be done in a streaming
/// manner.
pub struct StreamingSliceNode {
    start_offset: usize,
    length: usize,
    stream_offset: usize,
}

impl StreamingSliceNode {
    pub fn new(start_offset: usize, length: usize) -> Self {
        Self {
            start_offset,
            length,
            stream_offset: 0,
        }
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
        if self.stream_offset >= self.start_offset + self.length || self.length == 0 {
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
            let stop_offset = self.start_offset + self.length;

            while let Ok(morsel) = recv.recv().await {
                let morsel = morsel.map(|df| {
                    let height = df.height();

                    // Calculate start/stop offsets within df and update global offset.
                    let relative_start_offset = self
                        .start_offset
                        .saturating_sub(self.stream_offset)
                        .min(height);
                    let relative_stop_offset =
                        stop_offset.saturating_sub(self.stream_offset).min(height);
                    self.stream_offset += height;

                    let new_height = relative_stop_offset.saturating_sub(relative_start_offset);
                    if new_height != height {
                        df.slice(relative_start_offset as i64, new_height)
                    } else {
                        df
                    }
                });

                // Technically not necessary, but it's nice to already tell the
                // source to stop producing more morsels as we won't be
                // interested in the results anyway.
                if self.stream_offset >= stop_offset {
                    morsel.source_token().stop();
                }

                if morsel.df().height() > 0 && send.send(morsel).await.is_err() {
                    break;
                }

                if self.stream_offset >= stop_offset {
                    break;
                }
            }

            Ok(())
        }))
    }
}
