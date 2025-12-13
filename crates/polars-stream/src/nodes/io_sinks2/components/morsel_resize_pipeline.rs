use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_utils::IdxSize;

use crate::async_primitives::connector;
use crate::morsel::Morsel;
use crate::nodes::io_sinks2::components::sink_morsel::SinkMorsel;
use crate::nodes::io_sinks2::components::size::RowCountAndSize;

/// Splits large morsels / combines small morsels.
pub struct MorselResizePipeline {
    pub empty_with_schema_df: DataFrame,
    pub ideal_morsel_size: RowCountAndSize,
    pub inflight_morsel_semaphore: Arc<tokio::sync::Semaphore>,
    pub morsel_rx: connector::Receiver<Morsel>,
    pub morsel_tx: connector::Sender<SinkMorsel>,
}

impl MorselResizePipeline {
    /// # Returns
    /// Returns how many rows were sent. If wrapped in `Err(_)`, indicates that the send channel
    /// closed prematurely.
    pub async fn run(self) -> Result<RowCountAndSize, RowCountAndSize> {
        let MorselResizePipeline {
            empty_with_schema_df,
            ideal_morsel_size,
            inflight_morsel_semaphore,
            mut morsel_rx,
            mut morsel_tx,
        } = self;

        let mut buffered_rows: DataFrame = empty_with_schema_df;
        let mut received_size: RowCountAndSize = RowCountAndSize::default();
        let mut sent_size: RowCountAndSize = RowCountAndSize::default();

        let mut flush = false;

        loop {
            let buffered_size = received_size.checked_sub(sent_size).unwrap();
            assert_eq!(
                buffered_size.num_rows,
                IdxSize::try_from(buffered_rows.height()).unwrap()
            );
            let num_rows_to_take = ideal_morsel_size.num_rows_takeable_from(buffered_size);

            if num_rows_to_take < buffered_size.num_rows
                || num_rows_to_take == ideal_morsel_size.num_rows
                || (num_rows_to_take > 0 && flush)
            {
                let morsel_permit = inflight_morsel_semaphore
                    .clone()
                    .acquire_owned()
                    .await
                    .unwrap();

                let (df, remaining) = buffered_rows.split_at(
                    #[allow(clippy::unnecessary_fallible_conversions)]
                    i64::try_from(num_rows_to_take).unwrap(),
                );

                if morsel_tx
                    .send(SinkMorsel::new(df, morsel_permit))
                    .await
                    .is_err()
                {
                    return Err(sent_size);
                };

                buffered_rows = remaining;

                let sent_size_delta = RowCountAndSize {
                    num_rows: num_rows_to_take,
                    #[allow(clippy::useless_conversion)]
                    num_bytes: u64::from(num_rows_to_take)
                        .saturating_mul(buffered_size.row_byte_size())
                        .min(received_size.num_bytes),
                };

                sent_size = sent_size.checked_add(sent_size_delta).unwrap();

                continue;
            }

            if flush {
                break;
            }

            let Ok(morsel) = morsel_rx.recv().await else {
                flush = true;
                continue;
            };

            let df = morsel.into_df();
            let morsel_size = RowCountAndSize::new_from_df(&df);

            received_size = received_size
                .checked_add(RowCountAndSize {
                    num_rows: morsel_size.num_rows,
                    num_bytes: std::cmp::min(
                        morsel_size.num_bytes,
                        u64::MAX - received_size.num_bytes,
                    ),
                })
                .unwrap();

            buffered_rows.vstack_mut_owned_unchecked(df);
        }

        assert_eq!(buffered_rows.height(), 0);
        assert_eq!(received_size.num_rows, sent_size.num_rows);

        Ok(received_size)
    }
}
