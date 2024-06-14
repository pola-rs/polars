use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use parking_lot::Mutex;
use polars_core::frame::DataFrame;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskScope};
use crate::async_primitives::pipe::{Receiver, Sender};
use crate::morsel::Morsel;

struct KMergeMorsel(Morsel, usize);

impl Eq for KMergeMorsel {}

impl Ord for KMergeMorsel {
    fn cmp(&self, other: &Self) -> Ordering {
        // Intentionally reverse order, BinaryHeap is a max-heap but we want the
        // smallest sequence number.
        other.0.seq().cmp(&self.0.seq())
    }
}

impl PartialOrd for KMergeMorsel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for KMergeMorsel {
    fn eq(&self, other: &Self) -> bool {
        self.0.seq() == other.0.seq()
    }
}

#[derive(Default)]
pub struct InMemorySink {
    per_pipe_morsels: Mutex<Vec<VecDeque<Morsel>>>,
}

impl ComputeNode for InMemorySink {
    fn initialize(&mut self, _num_pipelines: usize) {
        self.per_pipe_morsels.get_mut().clear();
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
            let mut morsels = VecDeque::new();
            while let Ok(mut morsel) = recv.recv().await {
                morsel.take_consume_token();
                morsels.push_back(morsel);
            }

            self.per_pipe_morsels.lock().push(morsels);
            Ok(())
        })
    }

    fn finalize(&mut self) -> PolarsResult<Option<DataFrame>> {
        // Do a K-way merge on the morsels based on sequence id.
        let mut per_pipe_morsels = core::mem::take(&mut *self.per_pipe_morsels.get_mut());
        let mut dataframes = Vec::with_capacity(per_pipe_morsels.iter().map(|p| p.len()).sum());

        let mut kmerge = BinaryHeap::new();
        for (pipe_idx, pipe) in per_pipe_morsels.iter_mut().enumerate() {
            if let Some(morsel) = pipe.pop_front() {
                kmerge.push(KMergeMorsel(morsel, pipe_idx));
            }
        }

        while let Some(KMergeMorsel(morsel, pipe_idx)) = kmerge.pop() {
            let seq = morsel.seq();
            dataframes.push(morsel.into_df());
            while let Some(new_morsel) = per_pipe_morsels[pipe_idx].pop_front() {
                if new_morsel.seq() == seq {
                    dataframes.push(new_morsel.into_df());
                } else {
                    kmerge.push(KMergeMorsel(new_morsel, pipe_idx));
                    break;
                }
            }
        }

        Ok(Some(accumulate_dataframes_vertical_unchecked(dataframes)))
    }
}
