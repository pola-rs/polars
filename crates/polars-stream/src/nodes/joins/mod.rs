use std::sync::LazyLock;

use crossbeam_queue::ArrayQueue;
use polars_core::POOL;
use polars_error::PolarsResult;
use polars_utils::itertools::Itertools;
use rayon::prelude::*;

use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::connector::{Receiver, connector};
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::pipe::RecvPort;

pub mod cross_join;
pub mod equi_join;
pub mod in_memory;
#[cfg(feature = "semi_anti_join")]
pub mod semi_anti_join;

static JOIN_SAMPLE_LIMIT: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("POLARS_JOIN_SAMPLE_LIMIT")
        .map(|limit| limit.parse().unwrap())
        .unwrap_or(10_000_000)
});

// If one side is this much bigger than the other side we'll always use the
// smaller side as the build side without checking cardinalities.
const LOPSIDED_SAMPLE_FACTOR: usize = 10;

// TODO: improve, generalize this, and move it away from here.
struct BufferedStream {
    morsels: ArrayQueue<Morsel>,
    post_buffer_offset: MorselSeq,
}

impl BufferedStream {
    pub fn new(morsels: Vec<Morsel>, start_offset: MorselSeq) -> Self {
        // Relabel so we can insert into parallel streams later.
        let mut seq = start_offset;
        let queue = ArrayQueue::new(morsels.len().max(1));
        for mut morsel in morsels {
            morsel.set_seq(seq);
            queue.push(morsel).unwrap();
            seq = seq.successor();
        }

        Self {
            morsels: queue,
            post_buffer_offset: seq,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.morsels.is_empty()
    }

    #[allow(clippy::needless_lifetimes)]
    pub fn reinsert<'s, 'env>(
        &'s self,
        num_pipelines: usize,
        recv_port: Option<RecvPort<'_>>,
        scope: &'s TaskScope<'s, 'env>,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) -> Option<Vec<Receiver<Morsel>>> {
        let receivers = if let Some(p) = recv_port {
            p.parallel().into_iter().map(Some).collect_vec()
        } else {
            (0..num_pipelines).map(|_| None).collect_vec()
        };

        let source_token = SourceToken::new();
        let mut out = Vec::new();
        for orig_recv in receivers {
            let (mut new_send, new_recv) = connector();
            out.push(new_recv);
            let source_token = source_token.clone();
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                // Act like an InMemorySource node until cached morsels are consumed.
                let wait_group = WaitGroup::default();
                loop {
                    let Some(mut morsel) = self.morsels.pop() else {
                        break;
                    };
                    morsel.replace_source_token(source_token.clone());
                    morsel.set_consume_token(wait_group.token());
                    if new_send.send(morsel).await.is_err() {
                        return Ok(());
                    }
                    wait_group.wait().await;
                    // TODO: Unfortunately we can't actually stop here without
                    // re-buffering morsels from the stream that comes after.
                    // if source_token.stop_requested() {
                    //     break;
                    // }
                }

                if let Some(mut recv) = orig_recv {
                    while let Ok(mut morsel) = recv.recv().await {
                        if source_token.stop_requested() {
                            morsel.source_token().stop();
                        }
                        morsel.set_seq(morsel.seq().offset_by(self.post_buffer_offset));
                        if new_send.send(morsel).await.is_err() {
                            break;
                        }
                    }
                }
                Ok(())
            }));
        }
        Some(out)
    }
}

impl Default for BufferedStream {
    fn default() -> Self {
        Self {
            morsels: ArrayQueue::new(1),
            post_buffer_offset: MorselSeq::default(),
        }
    }
}

impl Drop for BufferedStream {
    fn drop(&mut self) {
        POOL.install(|| {
            // Parallel drop as the state might be quite big.
            (0..self.morsels.len())
                .into_par_iter()
                .for_each(|_| drop(self.morsels.pop()));
        })
    }
}
