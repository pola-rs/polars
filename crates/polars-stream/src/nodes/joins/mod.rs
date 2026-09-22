use std::sync::Arc;

use crossbeam_queue::ArrayQueue;
use polars_async::executor::{JoinHandle, TaskPriority, TaskScope};
use polars_async::primitives::wait_group::WaitGroup;
use polars_core::frame::DataFrame;
use polars_core::runtime::RAYON;
use polars_error::PolarsResult;
use polars_ooc::{MostRecentSpillContext, SpillFrame};
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::relaxed_cell::RelaxedCell;
use rayon::prelude::*;

use crate::morsel::{Morsel, MorselSeq, SourceToken, get_ideal_morsel_size};
use crate::pipe::{PortReceiver, PortSender, RecvPort, port_channel};

#[cfg(feature = "asof_join")]
pub mod asof_join;
pub mod cross_join;
pub mod equi_join;
pub mod in_memory;
pub mod merge_join;
#[cfg(feature = "iejoin")]
pub mod range_join;
mod runtime_filter;
#[cfg(feature = "semi_anti_join")]
pub mod semi_anti_join;
mod utils;

// If one side is this much bigger than the other side we'll always use the
// smaller side as the build side without checking cardinalities.
const LOPSIDED_SAMPLE_FACTOR: usize = 10;

/// Buffers the morsels of one side of a join until it ends, the sample limit
/// is reached, or the other side ended and this side has many times its rows.
async fn sample_sink(
    mut recv: PortReceiver,
    morsels: &mut Vec<Morsel>,
    len: &mut usize,
    this_final_len: Arc<RelaxedCell<usize>>,
    other_final_len: Arc<RelaxedCell<usize>>,
    join_sample_limit: usize,
) -> PolarsResult<()> {
    while let Ok(mut morsel) = recv.recv().await {
        *len += morsel.height();
        if *len >= join_sample_limit
            || *len
                >= other_final_len
                    .load()
                    .saturating_mul(LOPSIDED_SAMPLE_FACTOR)
        {
            morsel.source_token().stop();
        }

        drop(morsel.take_consume_token());
        morsels.push(morsel);
    }
    this_final_len.store(*len);
    Ok(())
}

/// Folds over the first `sample_limit` rows of the sampled morsels in
/// parallel, returning the reduced value and the number of rows folded.
fn fold_sample<T: Send>(
    morsels: &[Morsel],
    sample_limit: usize,
    init: impl Fn() -> T + Sync + Send,
    fold: impl Fn(T, &DataFrame) -> PolarsResult<T> + Sync + Send,
    reduce: impl Fn(T, T) -> T + Sync + Send,
) -> PolarsResult<(T, usize)> {
    let mut total_height = 0;
    let mut to_process_end = 0;
    while to_process_end < morsels.len() && total_height < sample_limit {
        total_height += morsels[to_process_end].height();
        to_process_end += 1;
    }
    if to_process_end == 0 {
        return Ok((init(), 0));
    }
    let last_morsel_idx = to_process_end - 1;
    let last_morsel_len = morsels[last_morsel_idx].height();
    let last_morsel_slice = last_morsel_len - total_height.saturating_sub(sample_limit);

    let out = RAYON.install(|| {
        morsels[..to_process_end]
            .par_iter()
            .enumerate()
            .try_fold(&init, |acc, (morsel_idx, morsel)| {
                let sliced;
                let pin_df = morsel.df_blocking();
                let df = if morsel_idx == last_morsel_idx {
                    sliced = pin_df.slice(0, last_morsel_slice);
                    &sliced
                } else {
                    &*pin_df
                };
                fold(acc, df)
            })
            .try_reduce(&init, |a, b| Ok(reduce(a, b)))
    })?;
    Ok((out, total_height.min(sample_limit)))
}

/// The rows per morsel to send `total_len` rows in morsels of about the ideal
/// size, with at least one morsel per pipeline.
fn emit_morsel_size(total_len: usize, num_pipelines: usize) -> usize {
    let ideal_morsel_count = (total_len / get_ideal_morsel_size()).max(1);
    let morsel_count = ideal_morsel_count.next_multiple_of(num_pipelines);
    total_len.div_ceil(morsel_count).max(1)
}

/// Sends the frames `next` produces one at a time, waiting for each to be
/// consumed. Stops when the receiver is gone or asks to stop.
async fn send_frames(
    mut send: PortSender,
    seq: &mut MorselSeq,
    mut next: impl FnMut() -> Option<DataFrame>,
) -> PolarsResult<()> {
    let wait_group = WaitGroup::default();
    let source_token = SourceToken::new();
    while let Some(df) = next() {
        let mut morsel = Morsel::new_unregistered(df, *seq, source_token.clone());
        *seq = seq.successor();
        morsel.set_consume_token(wait_group.token());
        if send.send(morsel).await.is_err() {
            return Ok(());
        }

        wait_group.wait().await;
        if source_token.stop_requested() {
            return Ok(());
        }
    }
    Ok(())
}

// TODO: improve, generalize this, and move it away from here.
struct BufferedStream {
    morsels: ArrayQueue<(SpillFrame, MorselSeq)>,
    post_buffer_offset: MorselSeq,
    _spill_ctx: Option<MostRecentSpillContext>,
}

impl BufferedStream {
    pub fn new(name: PlSmallStr, morsels: Vec<Morsel>, start_offset: MorselSeq) -> Self {
        // Relabel so we can insert into parallel streams later.
        let mut seq = start_offset;
        let ctx = MostRecentSpillContext::new(name);
        let queue = ArrayQueue::new(morsels.len().max(1));
        for morsel in morsels {
            let sf = SpillFrame::new_blocking(morsel.into_df_blocking(), &ctx);
            queue.push((sf, seq)).unwrap();
            seq = seq.successor();
        }

        Self {
            morsels: queue,
            post_buffer_offset: seq,
            _spill_ctx: Some(ctx),
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
    ) -> Option<Vec<PortReceiver>> {
        let receivers = if let Some(p) = recv_port {
            p.parallel().into_iter().map(Some).collect_vec()
        } else {
            (0..num_pipelines).map(|_| None).collect_vec()
        };

        let source_token = SourceToken::new();
        let mut out = Vec::new();
        for orig_recv in receivers {
            let (mut new_send, new_recv) = port_channel(None);
            out.push(new_recv);
            let source_token = source_token.clone();
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                // Act like an InMemorySource node until cached morsels are consumed.
                let wait_group = WaitGroup::default();
                loop {
                    let Some((sf, seq)) = self.morsels.pop() else {
                        break;
                    };
                    let mut morsel = Morsel::new(sf, seq, source_token.clone());
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
            _spill_ctx: None,
        }
    }
}

impl Drop for BufferedStream {
    fn drop(&mut self) {
        RAYON.install(|| {
            // Parallel drop as the state might be quite big.
            (0..self.morsels.len()).into_par_iter().for_each(|_| {
                drop(self.morsels.pop());
            });
        })
    }
}
