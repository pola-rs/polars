use std::collections::VecDeque;
use std::sync::Arc;

use polars_async::primitives::connector;
use polars_async::primitives::wait_group::WaitToken;
use polars_core::frame::DataFrame;
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_utils::IdxSize;
use polars_utils::calc_morsel_split::PartSizesIter;
use polars_utils::index::idxsize_to_u64;

use crate::morsel::Morsel;
use crate::nodes::io_sinks::components::df_with_offset::DfWithOffset;
use crate::nodes::io_sinks::components::sink_morsel::SinkMorsel;
use crate::nodes::io_sinks::components::size::{RowCountAndSize, TargetSinkMorselSize};

/// Splits large morsels / combines small morsels.
pub struct MorselResizePipeline {
    pub schema: Arc<Schema>,
    pub target_sink_morsel_size: TargetSinkMorselSize,
    pub inflight_morsel_semaphore: Arc<tokio::sync::Semaphore>,
    pub morsel_rx: connector::Receiver<Morsel>,
    pub morsel_tx: connector::Sender<SinkMorsel>,
}

impl MorselResizePipeline {
    /// # Returns
    /// Returns an error if number of rows overflowed `IdxSize::MAX`.
    pub async fn run(self) -> PolarsResult<RowCountAndSize> {
        let MorselResizePipeline {
            schema,
            target_sink_morsel_size,
            inflight_morsel_semaphore,
            mut morsel_rx,
            mut morsel_tx,
        } = self;

        let mut buffered_rows: VecDeque<DfWithOffset> = VecDeque::with_capacity(4);
        // <= physical_received_size, incremented as we scan more chunks from the physically
        // received morsel.
        let mut logical_received_size: RowCountAndSize = RowCountAndSize::default();
        let mut physical_received_size: RowCountAndSize = RowCountAndSize::default();
        // Must always be <= logical_received_size.
        let mut sent_size: RowCountAndSize = RowCountAndSize::default();

        let mut num_rows_to_take_iter = PartSizesIter::default();
        // Note: Lengths are reversed as we pop from this.
        let mut next_chunk_lengths: Vec<usize> = vec![];
        let mut wait_token: Option<WaitToken> = None;

        loop {
            let df_to_send = if let Some(num_rows_to_take) = num_rows_to_take_iter.next() {
                take_n_rows_from_buffered(Arc::clone(&schema), &mut buffered_rows, num_rows_to_take)
            } else if let Some(n_rows) = next_chunk_lengths.pop() {
                let next_chunk_size = logical_received_size
                    .calc_delta(n_rows as IdxSize, physical_received_size)
                    .unwrap();

                let logical_buffered_size = logical_received_size.checked_sub(sent_size).unwrap();

                let send_buffered_as_one_df;

                (send_buffered_as_one_df, num_rows_to_take_iter) = target_sink_morsel_size
                    .calc_next_splits(logical_buffered_size, next_chunk_size);

                logical_received_size = logical_received_size.add(next_chunk_size)?;

                if send_buffered_as_one_df {
                    take_n_rows_from_buffered(
                        Arc::clone(&schema),
                        &mut buffered_rows,
                        idxsize_to_u64(logical_buffered_size.num_rows),
                    )
                } else {
                    continue;
                }
            } else if let Ok(morsel) = morsel_rx.recv().await {
                assert_eq!(
                    logical_received_size.num_rows,
                    physical_received_size.num_rows
                );
                // Force num_bytes to match.
                logical_received_size = physical_received_size;

                let df: DataFrame;
                drop(wait_token.take());

                (df, _, _, wait_token) = morsel.into_inner();

                if df.height() == 0 {
                    continue;
                }

                let morsel_size = RowCountAndSize::new_from_df(&df);
                physical_received_size = physical_received_size.add(morsel_size)?;

                if let Some(s) = df
                    .columns()
                    .iter()
                    .filter_map(|c| c.as_series())
                    .max_by_key(|s| s.n_chunks())
                {
                    next_chunk_lengths.extend(s.chunk_lengths().rev());
                } else {
                    next_chunk_lengths.push(df.height());
                }

                buffered_rows.push_back(DfWithOffset::new(df));

                continue;
            } else {
                assert_eq!(
                    logical_received_size.num_rows,
                    physical_received_size.num_rows
                );
                // Force num_bytes to match.
                logical_received_size = physical_received_size;

                let logical_buffered_size = logical_received_size.checked_sub(sent_size).unwrap();

                if logical_buffered_size.num_rows == 0 {
                    assert!(buffered_rows.iter().all(|df| df.height() == 0));
                    break;
                }

                take_n_rows_from_buffered(
                    Arc::clone(&schema),
                    &mut buffered_rows,
                    idxsize_to_u64(logical_buffered_size.num_rows),
                )
            };

            let morsel_permit = inflight_morsel_semaphore
                .clone()
                .acquire_owned()
                .await
                .unwrap();

            sent_size = sent_size
                .add_delta(df_to_send.height() as IdxSize, logical_received_size)
                .unwrap();

            if morsel_tx
                .send(SinkMorsel::new(df_to_send, morsel_permit))
                .await
                .is_err()
            {
                return Ok(sent_size);
            };
        }

        assert!(buffered_rows.iter().all(|df| df.height() == 0));
        assert_eq!(logical_received_size.num_rows, sent_size.num_rows);
        assert_eq!(physical_received_size.num_rows, sent_size.num_rows);

        Ok(physical_received_size)
    }
}

/// # Panics
/// Panics if `n_rows` is greater than the total number of rows in `buffered_rows`.
fn take_n_rows_from_buffered(
    schema: Arc<Schema>,
    buffered_rows: &mut VecDeque<DfWithOffset>,
    mut n_rows: u64,
) -> DataFrame {
    if n_rows == 0 {
        return DataFrame::empty_with_arc_schema(schema);
    }

    let mut stacked_df: Option<DataFrame> = None;

    while n_rows != 0 {
        let next_df = buffered_rows.front_mut().unwrap();

        let next_df = if next_df.height() as u64 > n_rows {
            next_df.split_off_front(n_rows as usize)
        } else {
            buffered_rows.pop_front().unwrap().into_df()
        };

        if next_df.height() as u64 == n_rows && stacked_df.is_none() {
            return next_df;
        }

        let next_df_height = next_df.height();

        match stacked_df.as_mut() {
            Some(df) => {
                df.vstack_mut_owned_unchecked(next_df);
            },
            None => stacked_df = Some(next_df),
        };

        n_rows -= next_df_height as u64;
    }

    stacked_df.unwrap()
}
