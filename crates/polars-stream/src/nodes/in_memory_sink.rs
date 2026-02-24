use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use polars_core::schema::Schema;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;

use super::compute_node_prelude::*;
use crate::memory::MemoryTracker;
use crate::utils::in_memory_linearize::linearize;

pub struct InMemorySinkNode {
    morsels_per_pipe: Mutex<Vec<Vec<(MorselSeq, DataFrame)>>>,
    schema: Arc<Schema>,
    tracked_bytes: AtomicU64,
    memory_tracker: Option<Arc<MemoryTracker>>,
}

impl InMemorySinkNode {
    pub fn new(schema: Arc<Schema>) -> Self {
        Self {
            morsels_per_pipe: Mutex::default(),
            schema,
            tracked_bytes: AtomicU64::new(0),
            memory_tracker: None,
        }
    }

    /// Release all tracked memory back to the memory tracker.
    fn free_tracked_memory(&mut self) {
        if let Some(tracker) = &self.memory_tracker {
            let bytes = self.tracked_bytes.load(Ordering::Relaxed);
            if bytes > 0 {
                tracker.free(bytes);
                self.tracked_bytes.store(0, Ordering::Relaxed);
            }
        }
    }
}

impl ComputeNode for InMemorySinkNode {
    fn name(&self) -> &str {
        "in-memory-sink"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(send.is_empty());
        assert!(recv.len() == 1);

        // We are always ready to receive, unless the sender is done, then we're
        // also done.
        if recv[0] != PortState::Done {
            recv[0] = PortState::Ready;
        }
        Ok(())
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        true
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1 && send_ports.is_empty());
        let receivers = recv_ports[0].take().unwrap().parallel();

        // Store the memory tracker reference if a limit is configured.
        if state.memory_tracker.has_limit() && self.memory_tracker.is_none() {
            self.memory_tracker = Some(state.memory_tracker.clone());
        }

        for mut recv in receivers {
            let slf = &*self;
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let mut morsels = Vec::new();
                while let Ok(mut morsel) = recv.recv().await {
                    morsel.take_consume_token();
                    let seq = morsel.seq();
                    let df = morsel.into_df();

                    // Track memory usage and apply backpressure if needed.
                    if let Some(tracker) = &slf.memory_tracker {
                        let size = df.estimated_size() as u64;
                        tracker.alloc(size);
                        slf.tracked_bytes.fetch_add(size, Ordering::Relaxed);
                        tracker.wait_for_available().await;
                    }

                    morsels.push((seq, df));
                }

                slf.morsels_per_pipe.lock().push(morsels);
                Ok(())
            }));
        }
    }

    fn get_output(&mut self) -> PolarsResult<Option<DataFrame>> {
        self.free_tracked_memory();
        let morsels_per_pipe = core::mem::take(&mut *self.morsels_per_pipe.get_mut());
        let dataframes = linearize(morsels_per_pipe);
        if dataframes.is_empty() {
            Ok(Some(DataFrame::empty_with_schema(&self.schema)))
        } else {
            Ok(Some(accumulate_dataframes_vertical_unchecked(dataframes)))
        }
    }
}
