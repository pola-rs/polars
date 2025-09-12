use std::sync::Arc;

use polars_error::PolarsResult;
use polars_utils::slice_enum::Slice;

use crate::async_executor::{self, AbortOnDropHandle, TaskPriority};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::morsel_linearizer::MorselLinearizer;
use crate::morsel::Morsel;
use crate::nodes::io_sources::multi_scan::components::apply_extra_ops::ApplyExtraOps;
use crate::nodes::io_sources::multi_scan::components::row_counter::RowCounter;
use crate::nodes::io_sources::multi_scan::reader_interface::output::FileReaderOutputRecv;

pub struct PostApplyExtraOps {
    pub reader_output_port: FileReaderOutputRecv,
    pub ops_applier: Arc<ApplyExtraOps>,
    /// We have this because initialization of `ops_applier` causes `reader_output_port` to have the
    /// first morsel consumed.
    pub first_morsel: Morsel,
    pub first_morsel_position: RowCounter,
    pub num_pipelines: usize,
}

impl PostApplyExtraOps {
    pub fn run(self) -> (MorselLinearizer, AbortOnDropHandle<PolarsResult<()>>) {
        let PostApplyExtraOps {
            mut reader_output_port,
            ops_applier,
            first_morsel,
            first_morsel_position,
            num_pipelines,
        } = self;

        let (mut distr_tx, distr_receivers) = distributor_channel(num_pipelines, 1);

        // Distributor
        {
            let ops_applier = ops_applier.clone();
            async_executor::spawn(TaskPriority::Low, async move {
                // Position tracking
                let mut row_counter: RowCounter = first_morsel_position;

                let mut morsel = first_morsel;

                // Should only run the pipeline if we have an operation we need to apply.
                let ApplyExtraOps::Initialized {
                    physical_pre_slice,
                    external_filter_mask,
                    ..
                } = ops_applier.as_ref()
                else {
                    unreachable!();
                };

                assert!(
                    physical_pre_slice
                        .as_ref()
                        .is_none_or(|x| matches!(x, Slice::Positive { .. }))
                );

                loop {
                    let row_count_this_morsel = {
                        let physical_rows = morsel.df().height();
                        // # Multiple cases
                        // * If row deletions are being done in post-apply, we'll have the deleted row count here.
                        // * If row deletions were pushed to the reader, `external_filter_mask` here is `None`, so we'll
                        //   have 0 deleted rows in the `deleted_rows` counter.
                        //   * Instead, `physical_rows` will be a counter that has the `deleted_rows` count subtracted
                        //     from it (because we are taking the height of the morsels after the rows are deleted).
                        let deleted_rows = external_filter_mask.as_ref().map_or(0, |mask| {
                            let Slice::Positive { offset, len } = Slice::Positive {
                                offset: row_counter.num_physical_rows(),
                                len: morsel.df().height(),
                            }
                            .restrict_to_bounds(mask.len()) else {
                                unreachable!()
                            };

                            mask.slice(offset, len).num_deleted_rows()
                        });

                        RowCounter::new(physical_rows, deleted_rows)
                    };

                    // We hit this if a reader does not support PRE_SLICE.
                    if physical_pre_slice
                        .clone()
                        .is_some_and(|x| x.offsetted(row_counter.num_physical_rows()).len() == 0)
                    {
                        // Note: We do not return any flag indicating that we have reached end of slice
                        // from this context. The read should be stopped on a higher level by using
                        // the `row_position_on_end_tx` callback from the reader.
                        break;
                    }

                    if distr_tx.send((morsel, row_counter)).await.is_err() {
                        break;
                    }

                    row_counter = row_counter.add(row_count_this_morsel);

                    let Ok(v) = reader_output_port.recv().await else {
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
}
