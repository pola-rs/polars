use std::sync::Arc;

use polars_error::PolarsResult;
use polars_utils::IdxSize;

use crate::async_executor::{self, AbortOnDropHandle, TaskPriority};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::morsel_linearizer::MorselLinearizer;
use crate::morsel::Morsel;
use crate::nodes::io_sources::multi_file_reader::extra_ops::apply::ApplyExtraOps;
use crate::nodes::io_sources::multi_file_reader::reader_interface::output::FileReaderOutputRecv;

pub fn spawn_post_apply_pipeline(
    mut reader_port: FileReaderOutputRecv,
    ops_applier: Arc<ApplyExtraOps>,
    // We have this because initialization of `ops_applier` causes `reader_port` to have the
    // first morsel consumed.
    first_morsel: Morsel,
    num_pipelines: usize,
) -> (MorselLinearizer, AbortOnDropHandle<PolarsResult<()>>) {
    let (mut distr_tx, distr_receivers) = distributor_channel(num_pipelines, 1);

    // Distributor
    {
        let ops_applier = ops_applier.clone();
        async_executor::spawn(TaskPriority::Low, async move {
            // Number of rows received from this reader.
            let mut n_rows_received: IdxSize = 0;

            let mut morsel = first_morsel;

            // Should only run the pipeline if we have an operation we need to apply.
            let ApplyExtraOps::Initialized { pre_slice, .. } = ops_applier.as_ref() else {
                unreachable!();
            };

            loop {
                let h = morsel.df().height();
                let h = IdxSize::try_from(h).unwrap_or(IdxSize::MAX);

                // We hit this if a reader does not support PRE_SLICE.
                if pre_slice.clone().is_some_and(|x| {
                    x.offsetted(usize::try_from(n_rows_received).unwrap()).len() == 0
                }) {
                    // Note: We do not return any flag indicating that we have reached end of slice
                    // from this context. The read should be stopped on a higher level by using
                    // the `row_position_on_end_tx` callback from the reader.
                    break;
                }

                if distr_tx.send((morsel, n_rows_received)).await.is_err() {
                    break;
                }

                n_rows_received = n_rows_received.saturating_add(h);

                let Ok(v) = reader_port.recv().await else {
                    break;
                };

                morsel = v;
            }
        })
    };

    let (rx, senders) = MorselLinearizer::new(num_pipelines, 4);

    let worker_handles = distr_receivers
        .into_iter()
        .zip(senders)
        .map(|(mut morsel_rx, mut morsel_tx)| {
            let ops_applier = ops_applier.clone();

            AbortOnDropHandle::new(async_executor::spawn(TaskPriority::Low, async move {
                while let Ok((mut morsel, row_offset)) = morsel_rx.recv().await {
                    ops_applier.apply_to_df(morsel.df_mut(), row_offset)?;

                    if morsel_tx.insert(morsel).await.is_err() {
                        break;
                    }
                }

                PolarsResult::Ok(())
            }))
        })
        .collect::<Vec<_>>();

    let handle = AbortOnDropHandle::new(async_executor::spawn(TaskPriority::Low, async move {
        for handle in worker_handles {
            handle.await?;
        }

        Ok(())
    }));

    (rx, handle)
}
