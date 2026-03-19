use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_error::PolarsResult;
use polars_utils::IdxSize;

use crate::async_primitives::connector;
use crate::morsel::Morsel;
use crate::nodes::io_sinks::components::sink_morsel::SinkMorsel;
use crate::nodes::io_sinks::components::size::{RowCountAndSize, TakeableRowsProvider};

/// Splits large morsels / combines small morsels.
pub struct MorselResizePipeline {
    pub empty_with_schema_df: DataFrame,
    pub takeable_rows_provider: TakeableRowsProvider,
    pub inflight_morsel_semaphore: Arc<tokio::sync::Semaphore>,
    pub morsel_rx: connector::Receiver<Morsel>,
    pub morsel_tx: connector::Sender<SinkMorsel>,
}

impl MorselResizePipeline {
    /// # Returns
    /// Returns an error if number of rows overflowed `IdxSize::MAX`.
    pub async fn run(self) -> PolarsResult<RowCountAndSize> {
        let MorselResizePipeline {
            empty_with_schema_df,
            takeable_rows_provider,
            inflight_morsel_semaphore,
            mut morsel_rx,
            mut morsel_tx,
        } = self;

        let mut buffered_rows: DataFrame = empty_with_schema_df;
        let mut received_size: RowCountAndSize = RowCountAndSize::default();
        // Must always be <= received_size.
        let mut sent_size: RowCountAndSize = RowCountAndSize::default();

        let mut flush = false;

        loop {
            let buffered_size = received_size.checked_sub(sent_size).unwrap();
            assert_eq!(
                buffered_size.num_rows,
                IdxSize::try_from(buffered_rows.height()).unwrap()
            );

            if let Some(num_rows_to_take) =
                takeable_rows_provider.num_rows_takeable_from(buffered_size, flush)
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
                    return Ok(sent_size);
                };

                buffered_rows = remaining;

                sent_size = sent_size
                    .add_delta(num_rows_to_take, received_size)
                    .unwrap();

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

            received_size = received_size.add(morsel_size)?;

            buffered_rows.vstack_mut_owned_unchecked(df);
        }

        assert_eq!(buffered_rows.height(), 0);
        assert_eq!(received_size.num_rows, sent_size.num_rows);

        Ok(received_size)
    }
}
