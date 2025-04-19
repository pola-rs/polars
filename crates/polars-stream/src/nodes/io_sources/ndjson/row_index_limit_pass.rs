use std::cmp::Reverse;

use polars_core::frame::DataFrame;
use polars_error::{PolarsResult, polars_bail};
use polars_io::RowIndex;
use polars_utils::IdxSize;
use polars_utils::priority::Priority;

use crate::async_primitives::linearizer::Linearizer;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::nodes::io_sources::multi_file_reader::reader_interface::output::FileReaderOutputSend;

pub struct ApplyRowIndexOrLimit {
    pub morsel_receiver: Linearizer<Priority<Reverse<MorselSeq>, DataFrame>>,
    pub morsel_tx: FileReaderOutputSend,
    pub limit: Option<usize>,
    pub row_index: Option<RowIndex>,
    pub verbose: bool,
}

impl ApplyRowIndexOrLimit {
    pub async fn run(self) -> PolarsResult<()> {
        let ApplyRowIndexOrLimit {
            mut morsel_receiver,
            mut morsel_tx,
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

            // No wait group logic here, already attached by line batch processors.

            if morsel_tx
                .send_morsel(Morsel::new(df, morsel_seq, SourceToken::new()))
                .await
                .is_err()
            {
                break;
            }

            if limit.is_some_and(|x| n_rows_received >= x) {
                break;
            }
        }

        // Explicit drop to stop NDJSON parsing as soon as possible.
        drop(morsel_receiver);

        if verbose {
            eprintln!("[NDJSON ApplyRowIndexOrLimit]: returning");
        }

        Ok(())
    }
}
