use std::sync::Arc;

use parking_lot::Mutex;
use polars_core::schema::Schema;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;

use super::compute_node_prelude::*;
use crate::utils::in_memory_linearize::linearize;

pub struct InMemorySinkNode {
    morsels_per_pipe: Mutex<Vec<Vec<Morsel>>>,
    schema: Arc<Schema>,
}

impl InMemorySinkNode {
    pub fn new(schema: Arc<Schema>) -> Self {
        Self {
            morsels_per_pipe: Mutex::default(),
            schema,
        }
    }
}

impl ComputeNode for InMemorySinkNode {
    fn name(&self) -> &str {
        "in_memory_sink"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
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
        recv: &mut [Option<RecvPort<'_>>],
        send: &mut [Option<SendPort<'_>>],
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv.len() == 1 && send.is_empty());
        let receivers = recv[0].take().unwrap().parallel();

        for mut recv in receivers {
            let slf = &*self;
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let mut morsels = Vec::new();
                while let Ok(mut morsel) = recv.recv().await {
                    morsel.take_consume_token();
                    morsels.push(morsel);
                }

                slf.morsels_per_pipe.lock().push(morsels);
                Ok(())
            }));
        }
    }

    fn get_output(&mut self) -> PolarsResult<Option<DataFrame>> {
        let morsels_per_pipe = core::mem::take(&mut *self.morsels_per_pipe.get_mut());
        let dataframes = linearize(morsels_per_pipe);
        if dataframes.is_empty() {
            Ok(Some(DataFrame::empty_with_schema(&self.schema)))
        } else {
            Ok(Some(accumulate_dataframes_vertical_unchecked(dataframes)))
        }
    }
}
