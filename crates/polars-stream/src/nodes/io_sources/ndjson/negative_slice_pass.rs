use std::cmp::Reverse;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use polars_core::frame::DataFrame;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_error::{PolarsResult, polars_bail};
use polars_io::RowIndex;
use polars_utils::IdxSize;
use polars_utils::priority::Priority;

use crate::async_executor;
use crate::async_executor::AbortOnDropHandle;
use crate::async_primitives::connector;
use crate::async_primitives::linearizer::Linearizer;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::{Morsel, MorselSeq, get_ideal_morsel_size};
use crate::nodes::io_sources::MorselOutput;

/// Outputs a stream of morsels in reverse order from which they were received.
/// Attaches (properly offsetted) row index if necessary.
///
/// Used for negative slicing in NDJSON, where the morsels of the file are sent from back to front.
pub struct MorselStreamReverser {
    pub morsel_receiver: Linearizer<Priority<Reverse<MorselSeq>, DataFrame>>,
    pub phase_tx_receivers: Vec<connector::Receiver<MorselOutput>>,
    /// Slice from right to left.
    pub offset_len_rtl: (usize, usize),
    pub row_index: Option<(RowIndex, tokio::sync::oneshot::Receiver<usize>)>,
    pub verbose: bool,
}

impl MorselStreamReverser {
    pub async fn run(self) -> PolarsResult<()> {
        let MorselStreamReverser {
            mut morsel_receiver,
            phase_tx_receivers,
            offset_len_rtl,
            row_index,
            verbose,
        } = self;

        // Accumulated morsels
        let mut acc_morsels: Vec<(MorselSeq, DataFrame)> =
            Vec::with_capacity(phase_tx_receivers.len().clamp(16, 64));

        if verbose {
            eprintln!("MorselStreamReverser: start receiving");
        }

        let mut n_rows_received: usize = 0;

        while let Some(Priority(Reverse(morsel_seq), df)) = morsel_receiver.get().await {
            if acc_morsels.len() == acc_morsels.capacity() {
                let morsel_seq = acc_morsels.last().unwrap().0;
                let combined = combine_acc_morsels_reverse(&mut acc_morsels);
                acc_morsels.push((morsel_seq, combined));
            }

            n_rows_received = n_rows_received.saturating_add(df.height());

            acc_morsels.push((morsel_seq, df));

            // Note: The line batch distributor already skips the offset portion.
            if n_rows_received >= offset_len_rtl.1 {
                break;
            }
        }

        if verbose {
            eprintln!("MorselStreamReverser: dropping receiver");
        }

        drop(morsel_receiver);

        if acc_morsels.is_empty() {
            if verbose {
                eprintln!("MorselStreamReverser: no morsels, returning");
            }

            return Ok(());
        }

        // We don't assert height because the slice may overrun the file
        let combined_df = combine_acc_morsels_reverse(&mut acc_morsels);
        drop(acc_morsels);

        // The NDJSON workers don't stop at exactly the right number of rows (they stop when they
        // see the channel closed).
        let combined_df = if combined_df.height() > offset_len_rtl.1 {
            combined_df.slice(
                i64::try_from(combined_df.height() - offset_len_rtl.1).unwrap(),
                usize::MAX,
            )
        } else {
            combined_df
        };

        let row_index = if let Some((row_index, total_row_count_rx)) = row_index {
            if verbose {
                eprintln!("MorselStreamReverser: wait for total row count");
            }

            let Ok(total_count) = total_row_count_rx.await else {
                // Errored, or empty file.
                if verbose {
                    eprintln!("MorselStreamReverser: did not receive total row count, returning");
                }

                return Ok(());
            };

            if verbose {
                eprintln!("MorselStreamReverser: got total row count: {}", total_count);
            }

            // Convert to position from beginning
            // Note: We add the df height here rather than the slice length as the negative slice
            // could go past the start of the file.
            let n_from_end = offset_len_rtl.0 + combined_df.height();
            let n_from_start = total_count - n_from_end;

            if IdxSize::try_from(n_from_start)
                .ok()
                .and_then(|x| x.checked_add(row_index.offset))
                .is_none()
            {
                polars_bail!(
                    ComputeError:
                    "row_index with offset {} overflows at {} rows",
                    row_index.offset, n_from_start
                )
            }

            Some(RowIndex {
                name: row_index.name,
                offset: row_index.offset + n_from_start as IdxSize,
            })
        } else {
            None
        };

        let combined_df = Arc::new(combined_df);
        let chunk_size = get_ideal_morsel_size();
        let n_chunks = combined_df.height().div_ceil(chunk_size);
        let num_pipelines = phase_tx_receivers.len();
        let n_tasks = num_pipelines.min(n_chunks);
        let chunk_idx_arc = Arc::new(AtomicUsize::new(0));

        if verbose {
            eprintln!(
                "MorselStreamReverser: creating send tasks: \
                n_rows: {}, \
                n_chunks: {}, \
                chunk_size: {}, \
                num_pipelines: {} \
                n_tasks: {} \
                ",
                combined_df.height(),
                n_chunks,
                chunk_size,
                num_pipelines,
                n_tasks,
            );
        }

        // Otherwise we will wrap around on fetch_add
        assert!(usize::MAX - n_chunks >= n_tasks);

        let sender_join_handles = phase_tx_receivers
            .into_iter()
            .take(n_tasks)
            .map(|mut phase_tx_receiver| {
                let chunk_idx_arc = chunk_idx_arc.clone();
                let combined_df = combined_df.clone();
                let row_index = row_index.clone();
                AbortOnDropHandle::new(async_executor::spawn(
                    async_executor::TaskPriority::Low,
                    async move {
                        let Ok(mut morsel_output) = phase_tx_receiver.recv().await else {
                            return Ok(());
                        };

                        let wait_group = WaitGroup::default();

                        'outer: loop {
                            let chunk_idx =
                                chunk_idx_arc.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                            if chunk_idx >= n_chunks {
                                return Ok(());
                            }

                            let row_offset = chunk_idx.saturating_mul(chunk_size);
                            let mut df =
                                combined_df.slice(row_offset.try_into().unwrap(), chunk_size);

                            assert!(!df.is_empty()); // If we did our calculations properly

                            if let Some(row_index) = row_index.clone() {
                                let offset = row_index.offset.saturating_add(
                                    IdxSize::try_from(row_offset).unwrap_or(IdxSize::MAX),
                                );

                                if offset.checked_add(df.height() as IdxSize).is_none() {
                                    polars_bail!(
                                        ComputeError:
                                        "row_index with offset {} overflows at {} rows",
                                        row_index.offset, row_offset.saturating_add(df.height())
                                    )
                                };

                                unsafe {
                                    df.with_row_index_mut(row_index.name.clone(), Some(offset))
                                };
                            }

                            let mut morsel = Morsel::new(
                                df,
                                MorselSeq::new(chunk_idx as u64),
                                morsel_output.source_token.clone(),
                            );

                            morsel.set_consume_token(wait_group.token());

                            if morsel_output.port.send(morsel).await.is_err() {
                                break 'outer;
                            }

                            wait_group.wait().await;

                            if morsel_output.source_token.stop_requested() {
                                morsel_output.outcome.stop();
                                drop(morsel_output);

                                let Ok(next_output) = phase_tx_receiver.recv().await else {
                                    break;
                                };

                                morsel_output = next_output;
                            }
                        }

                        Ok(())
                    },
                ))
            })
            .collect::<Vec<_>>();

        for handle in sender_join_handles {
            handle.await?;
        }

        Ok(())
    }
}

/// # Panics
/// Panics if `acc_morsels` is empty.
fn combine_acc_morsels_reverse(acc_morsels: &mut Vec<(MorselSeq, DataFrame)>) -> DataFrame {
    // Morsel seq increasing order.
    debug_assert!(acc_morsels.windows(2).all(|x| {
        let &[(l, _), (r, _)] = x.try_into().unwrap();
        r > l
    }));

    accumulate_dataframes_vertical_unchecked(acc_morsels.drain(..).rev().map(|(_, df)| df))
}
