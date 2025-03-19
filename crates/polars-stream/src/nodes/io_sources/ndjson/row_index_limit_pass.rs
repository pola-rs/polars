use std::cmp::Reverse;

use polars_core::frame::DataFrame;
use polars_error::{PolarsResult, polars_bail};
use polars_io::RowIndex;
use polars_utils::IdxSize;
use polars_utils::priority::Priority;

use crate::async_executor::{AbortOnDropHandle, TaskPriority, spawn};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::async_primitives::wait_group::WaitGroup;
use crate::async_primitives::{connector, distributor_channel};
use crate::morsel::{Morsel, MorselSeq};
use crate::nodes::io_sources::MorselOutput;

pub struct ApplyRowIndexOrLimit {
    pub morsel_receiver: Linearizer<Priority<Reverse<MorselSeq>, DataFrame>>,
    pub phase_tx_receivers: Vec<connector::Receiver<MorselOutput>>,
    pub limit: Option<usize>,
    pub row_index: Option<RowIndex>,
    pub verbose: bool,
}

impl ApplyRowIndexOrLimit {
    pub async fn run(self) -> PolarsResult<()> {
        let ApplyRowIndexOrLimit {
            mut morsel_receiver,
            phase_tx_receivers,
            limit,
            row_index,
            verbose,
        } = self;

        debug_assert!(limit.is_some() || row_index.is_some());

        if verbose {
            eprintln!(
                "[NDJSON ApplyRowIndexOrLimit]: init: \
                limit: {:?} \
                row_index: {:?}",
                &limit, &row_index
            );
        }

        let (mut morsel_distributor, phase_sender_handles) =
            init_morsel_distributor(phase_tx_receivers);

        let mut n_rows_received: usize = 0;

        while let Some(Priority(Reverse(morsel_seq), mut df)) = morsel_receiver.get().await {
            if let Some(limit) = &limit {
                let remaining = *limit - n_rows_received;
                if remaining < df.height() {
                    df = df.slice(0, remaining);
                }
            }

            if let Some(row_index) = &row_index {
                let offset = row_index
                    .offset
                    .saturating_add(IdxSize::try_from(n_rows_received).unwrap_or(IdxSize::MAX));

                if offset.checked_add(df.height() as IdxSize).is_none() {
                    polars_bail!(
                        ComputeError:
                        "row_index with offset {} overflows at {} rows",
                        row_index.offset, n_rows_received.saturating_add(df.height())
                    )
                };

                unsafe { df.with_row_index_mut(row_index.name.clone(), Some(offset)) };
            }

            n_rows_received = n_rows_received.saturating_add(df.height());

            if morsel_distributor.send((morsel_seq, df)).await.is_err() {
                break;
            }

            if limit.is_some_and(|x| n_rows_received >= x) {
                break;
            }
        }

        // Explicit drop to stop NDJSON parsing as soon as possible.
        drop(morsel_receiver);
        drop(morsel_distributor);

        if verbose {
            eprintln!("[NDJSON ApplyRowIndexOrLimit]: wait for morsel distributor handles");
        }

        for handle in phase_sender_handles {
            handle.await?;
        }

        if verbose {
            eprintln!("[NDJSON ApplyRowIndexOrLimit]: returning");
        }

        Ok(())
    }
}

/// Initialize and connect a distributor to the morsel outputs. Returns the distributor and the join
/// handles for the tasks that send to the morsel outputs.
#[allow(clippy::type_complexity)]
fn init_morsel_distributor(
    phase_tx_receivers: Vec<connector::Receiver<MorselOutput>>,
) -> (
    distributor_channel::Sender<(MorselSeq, DataFrame)>,
    Vec<AbortOnDropHandle<PolarsResult<()>>>,
) {
    let (tx, dist_receivers) =
        distributor_channel::<(MorselSeq, DataFrame)>(phase_tx_receivers.len(), 1);

    let join_handles = phase_tx_receivers
        .into_iter()
        .zip(dist_receivers)
        .map(|(mut phase_tx_receiver, mut morsel_rx)| {
            AbortOnDropHandle::new(spawn(TaskPriority::Low, async move {
                let Ok(mut morsel_output) = phase_tx_receiver.recv().await else {
                    return Ok(());
                };

                let wait_group = WaitGroup::default();

                'outer: loop {
                    let Ok((morsel_seq, df)) = morsel_rx.recv().await else {
                        return Ok(());
                    };

                    let mut morsel =
                        Morsel::new(df, morsel_seq, morsel_output.source_token.clone());
                    morsel.set_consume_token(wait_group.token());

                    if morsel_output.port.send(morsel).await.is_err() {
                        break 'outer;
                    }

                    wait_group.wait().await;

                    if morsel_output.source_token.stop_requested() {
                        morsel_output.outcome.stop();
                        drop(morsel_output);

                        let Ok(next_output) = phase_tx_receiver.recv().await else {
                            break 'outer;
                        };

                        morsel_output = next_output;
                    }
                }

                Ok(())
            }))
        })
        .collect();

    (tx, join_handles)
}
