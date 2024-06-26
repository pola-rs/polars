use std::sync::Arc;

use parking_lot::Mutex;
use polars_core::frame::DataFrame;
use polars_core::schema::Schema;
use polars_core::series::Series;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskScope};
use crate::async_primitives::pipe::{Receiver, Sender};
use crate::morsel::Morsel;
use crate::utils::in_memory_linearize::linearize;

pub struct InMemorySink {
    morsels_per_pipe: Mutex<Vec<Vec<Morsel>>>,
    schema: Arc<Schema>,
}

impl InMemorySink {
    pub fn new(schema: Arc<Schema>) -> Self {
        Self {
            morsels_per_pipe: Mutex::default(),
            schema,
        }
    }
}

impl ComputeNode for InMemorySink {
    fn initialize(&mut self, _num_pipelines: usize) {
        self.morsels_per_pipe.get_mut().clear();
    }

    fn spawn<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        _pipeline: usize,
        recv: Vec<Receiver<Morsel>>,
        send: Vec<Sender<Morsel>>,
        _state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>> {
        assert!(send.is_empty());
        let [mut recv] = <[_; 1]>::try_from(recv).ok().unwrap();

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
        let morsels_per_pipe = core::mem::take(&mut *self.morsels_per_pipe.get_mut());
        let dataframes = linearize(morsels_per_pipe);
        if dataframes.is_empty() {
            Ok(Some(DataFrame::empty_with_schema(&self.schema)))
        } else {
            Ok(Some(accumulate_dataframes_vertical_unchecked(dataframes)))
        }
    }
}
