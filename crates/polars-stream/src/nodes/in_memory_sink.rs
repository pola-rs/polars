use std::sync::Arc;

use parking_lot::Mutex;
use polars_core::frame::DataFrame;
use polars_core::schema::Schema;
use polars_core::series::Series;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;

use super::{ComputeNode, PortState};
use crate::async_executor::{JoinHandle, TaskScope};
use crate::async_primitives::pipe::{Receiver, Sender};
use crate::morsel::Morsel;
use crate::utils::in_memory_linearize::linearize;

pub struct InMemorySinkNode {
    morsels_per_pipe: Mutex<Vec<Vec<Morsel>>>,
    schema: Arc<Schema>,
    done: bool,
}

impl InMemorySinkNode {
    pub fn new(schema: Arc<Schema>) -> Self {
        Self {
            morsels_per_pipe: Mutex::default(),
            schema,
            done: false,
        }
    }
}

impl ComputeNode for InMemorySinkNode {
    fn name(&self) -> &'static str {
        "in_memory_sink"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) {
        assert!(send.is_empty());
        assert!(recv.len() == 1);

        // If a sink is done, it's done, otherwise it will just reflect its
        // input state.
        if self.done {
            recv[0] = PortState::Done;
        }
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        !self.done
    }

    fn initialize(&mut self, _num_pipelines: usize) {
        self.morsels_per_pipe.get_mut().clear();
    }

    fn spawn<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        _pipeline: usize,
        recv: &mut [Option<Receiver<Morsel>>],
        send: &mut [Option<Sender<Morsel>>],
        _state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>> {
        assert!(recv.len() == 1 && send.is_empty());
        let mut recv = recv[0].take().unwrap();

        scope.spawn_task(true, async move {
            let mut morsels = Vec::new();
            while let Ok(mut morsel) = recv.recv().await {
                morsel.take_consume_token();
                morsels.push(morsel);
            }

            self.morsels_per_pipe.lock().push(morsels);
            Ok(())
        })
    }

    fn finalize(&mut self) -> PolarsResult<Option<DataFrame>> {
        self.done = true;

        let morsels_per_pipe = core::mem::take(&mut *self.morsels_per_pipe.get_mut());
        let dataframes = linearize(morsels_per_pipe);
        if dataframes.is_empty() {
            Ok(Some(DataFrame::empty_with_schema(&self.schema)))
        } else {
            Ok(Some(accumulate_dataframes_vertical_unchecked(dataframes)))
        }
    }
}
