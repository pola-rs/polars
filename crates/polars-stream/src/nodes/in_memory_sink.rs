use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use parking_lot::Mutex;
use polars_core::frame::DataFrame;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_core::utils::rayon::iter::{IntoParallelIterator, ParallelIterator};
use polars_core::POOL;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_utils::priority::Priority;
use polars_utils::sync::SyncPtr;

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskScope};
use crate::async_primitives::pipe::{Receiver, Sender};
use crate::morsel::Morsel;
use crate::utils::in_memory_linearize::linearize;

#[derive(Default)]
pub struct InMemorySink {
    morsels_per_pipe: Mutex<Vec<Vec<Morsel>>>,
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
        let mut morsels_per_pipe = core::mem::take(&mut *self.morsels_per_pipe.get_mut());
        let dataframes = linearize(morsels_per_pipe);
        Ok(Some(accumulate_dataframes_vertical_unchecked(dataframes)))
    }
}
