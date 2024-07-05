use std::sync::Arc;

use parking_lot::Mutex;

use super::compute_node_prelude::*;
use crate::async_primitives::distributor_channel::{
    distributor_channel, Receiver as DistrReceiver,
};
use crate::utils::linearizer::{Inserter, Linearizer};
use crate::{DEFAULT_DISTRIBUTOR_BUFFER_SIZE, DEFAULT_LINEARIZER_BUFFER_SIZE};

#[derive(Copy, Clone, Default)]
struct GlobalState {
    stream_offset: usize,
    morsel_seq: MorselSeq,
}

/// A node that will pass-through up to length rows, starting at start_offset.
/// Since start_offset must be non-negative this can be done in a streaming
/// manner.
pub struct StreamingSliceNode {
    start_offset: usize,
    length: usize,

    global_state: Mutex<GlobalState>,

    num_pipelines: usize,
    #[allow(clippy::type_complexity)]
    per_pipeline_resources: Mutex<Vec<Option<(Inserter, DistrReceiver<Morsel>)>>>,
}

impl StreamingSliceNode {
    pub fn new(start_offset: usize, length: usize) -> Self {
        Self {
            start_offset,
            length,
            global_state: Mutex::default(),
            num_pipelines: 0,
            per_pipeline_resources: Mutex::default(),
        }
    }
}

impl ComputeNode for StreamingSliceNode {
    fn name(&self) -> &str {
        "streaming_slice"
    }

    fn initialize(&mut self, num_pipelines: usize) {
        self.num_pipelines = num_pipelines;
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) {
        let global_state = self.global_state.lock();
        if global_state.stream_offset >= self.start_offset + self.length || self.length == 0 {
            recv[0] = PortState::Done;
            send[0] = PortState::Done;
        } else {
            recv.swap_with_slice(send);
        }
    }

    fn spawn_global<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        state: &'s ExecutionState,
    ) -> Option<JoinHandle<PolarsResult<()>>> {
        let (mut linearizer, inserters) =
            Linearizer::new(self.num_pipelines, DEFAULT_LINEARIZER_BUFFER_SIZE);
        let (mut sender, receivers) =
            distributor_channel(self.num_pipelines, DEFAULT_DISTRIBUTOR_BUFFER_SIZE);
        {
            let per_pipeline_resources = &mut *self.per_pipeline_resources.lock();
            per_pipeline_resources.clear();
            per_pipeline_resources.extend(inserters.into_iter().zip(receivers).map(Some));
        }

        Some(scope.spawn_task(TaskPriority::High, async move {
            let mut global_state = *self.global_state.lock();
            let stop_offset = self.start_offset + self.length;

            while let Some(morsel) = linearizer.get().await {
                let mut df = morsel.into_df();
                let height = df.height();

                // Start/stop offsets within df.
                let relative_start_offset = self
                    .start_offset
                    .saturating_sub(global_state.stream_offset)
                    .min(height);
                let relative_stop_offset = stop_offset
                    .saturating_sub(global_state.stream_offset)
                    .min(height);
                if relative_start_offset < relative_stop_offset {
                    let new_height = relative_stop_offset - relative_start_offset;
                    if new_height != height {
                        df = df.slice(relative_start_offset as i64, new_height);
                    }
                    sender.send(Morsel::new(df, global_state.morsel_seq)).await;
                    global_state.morsel_seq = global_state.morsel_seq.successor();
                }

                global_state.stream_offset += height;
                if global_state.stream_offset >= stop_offset {
                    break;
                }
            }

            *self.global_state.lock() = global_state;
            Ok(())
        }))
    }

    fn spawn<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        pipeline: usize,
        recv: &mut [Option<Receiver<Morsel>>],
        send: &mut [Option<Sender<Morsel>>],
        state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>> {
        assert!(recv.len() == 1 && send.len() == 1);
        let mut recv = recv[0].take().unwrap();
        let mut send = send[0].take().unwrap();
        let (mut inserter, mut distr_recv) =
            self.per_pipeline_resources.lock()[pipeline].take().unwrap();

        let insert_join = scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = recv.recv().await {
                if inserter.insert(morsel).await.is_err() {
                    break;
                }
            }

            PolarsResult::Ok(())
        });

        scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = distr_recv.recv().await {
                if send.send(morsel).await.is_err() {
                    break;
                }
            }

            insert_join.await?;
            Ok(())
        })
    }
}
