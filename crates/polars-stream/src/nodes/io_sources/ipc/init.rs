use std::sync::Arc;

use polars_error::PolarsResult;
use polars_utils::IdxSize;

use super::record_batch_data_fetch::RecordBatchDataFetcher;
use super::record_batch_decode::RecordBatchDecoder;
use super::{AsyncTaskData, IpcReadImpl};
use crate::async_executor;
use crate::morsel::{Morsel, SourceToken, get_ideal_morsel_size};
use crate::nodes::io_sources::ipc::ROW_COUNT_OVERFLOW_ERR;
use crate::nodes::io_sources::multi_scan::reader_interface::output::FileReaderOutputSend;
use crate::nodes::io_sources::parquet::init::split_to_morsels;
use crate::nodes::{MorselSeq, TaskPriority};
use crate::utils::tokio_handle_ext::AbortOnDropHandle;

impl IpcReadImpl {
    pub(crate) fn run(mut self) -> AsyncTaskData {
        dbg!("start run for IpcReadImpl");

        let verbose = self.verbose;

        if verbose {
            eprintln!("[IPCFileReader]: {:?}", &self.config);
            eprintln!(
                "[IPCFileReader]: record batch count: {:?}",
                self.metadata.blocks.len()
            );
        }

        let io_runtime = polars_io::pl_async::get_runtime();

        // Extract parameters.
        let metadata = self.metadata.clone();
        let record_batch_prefetch_size = self.config.record_batch_prefetch_size;
        let memory_prefetch_func = self.memory_prefetch_func;
        let ideal_morsel_size = get_ideal_morsel_size();

        if verbose {
            eprintln!("[IPCFileReader]: ideal_morsel_size: {ideal_morsel_size}");
        }

        let byte_source = self.byte_source.clone();

        let record_batch_decoder = self.init_record_batch_decoder();
        let record_batch_decoder = Arc::new(record_batch_decoder);

        // Set up channels.
        let (prefetch_send, mut prefetch_recv) =
            tokio::sync::mpsc::channel(record_batch_prefetch_size);
        let (decode_send, mut decode_recv) = tokio::sync::mpsc::channel(self.config.num_pipelines);
        let (mut morsel_send, morsel_recv) = FileReaderOutputSend::new_serial();

        // Fast-path for empty slice
        if self.slice_range.is_empty() {
            return (
                morsel_recv,
                AbortOnDropHandle(io_runtime.spawn(std::future::ready(Ok(())))),
            );
        }

        let n_rows_in_file_tx = self.n_rows_in_file_tx;
        let row_position_on_end_tx = self.row_position_on_end_tx;

        // Task: Prefetch.
        let prefetch_task = AbortOnDropHandle(io_runtime.spawn(async move {
            dbg!("start task: prefetch_task"); //kdn

            let mut record_batch_data_fetcher = RecordBatchDataFetcher {
                memory_prefetch_func,
                metadata,
                byte_source,
                record_batch_idx: 0,
            };

            // We fetch all record batches so that we know the total number of rows.
            // @TODO: In case of slicing, it would suffice to fetch the record batch
            // headers for any record batch that falls outside of the slice, or not
            // at all.
            while let Some(prefetch) = record_batch_data_fetcher.next().await {
                if prefetch_send.send(prefetch?).await.is_err() {
                    break;
                }
            }

            PolarsResult::Ok(())
        }));

        // Task: Decode.
        let decode_task = AbortOnDropHandle(io_runtime.spawn(async move {
            dbg!("start task: decode_task"); //kdn
            let mut current_row_offset: IdxSize = 0;

            while let Some(prefetch) = prefetch_recv.recv().await {
                let record_batch_data = prefetch.await.unwrap()?;

                // Fetch every record batch so we can determine total row count.
                let rb_num_rows = record_batch_data.num_rows;
                let rb_num_rows =
                    IdxSize::try_from(rb_num_rows).map_err(|_| ROW_COUNT_OVERFLOW_ERR)?;
                let row_range_end = current_row_offset
                    .checked_add(rb_num_rows)
                    .ok_or(ROW_COUNT_OVERFLOW_ERR)?;
                let row_range = current_row_offset..row_range_end;
                current_row_offset = row_range_end;

                // Only pass to decoder if we need the data.
                if (row_range.start as usize) < self.slice_range.end {
                    let record_batch_decoder = record_batch_decoder.clone();
                    let decode_fut = async_executor::spawn(TaskPriority::High, async move {
                        record_batch_decoder
                            .record_batch_data_to_df(record_batch_data, row_range)
                            .await
                    });
                    if decode_send.send(decode_fut).await.is_err() {
                        break;
                    }
                };
            }

            let current_row_offset =
                IdxSize::try_from(current_row_offset).map_err(|_| ROW_COUNT_OVERFLOW_ERR)?;

            // Handle callback.
            if let Some(row_position_on_end_tx) = row_position_on_end_tx {
                _ = row_position_on_end_tx.send(current_row_offset);
            }
            if let Some(n_rows_in_file_tx) = n_rows_in_file_tx {
                _ = n_rows_in_file_tx.send(current_row_offset); //kdn
            }

            PolarsResult::Ok(())
        }));

        // Task: Distributor.
        // Distributes morsels across pipelines. This does not perform any CPU or I/O bound work -
        // it is purely a dispatch loop. Run on the computational executor to reduce context switches.
        let last_morsel_min_split = self.config.num_pipelines;
        let distribute_task = async_executor::spawn(TaskPriority::High, async move {
            dbg!("start task: distribute_task"); //kdn
            let mut morsel_seq = MorselSeq::default();
            // Note: We don't use this (it is handled by the bridge). But morsels require a source token.
            let source_token = SourceToken::new();

            // Decode first non-empty morsel.
            let mut next = None;
            loop {
                let Some(decode_fut) = decode_recv.recv().await else {
                    break;
                };
                let df = decode_fut.await?;
                if df.height() == 0 {
                    continue;
                }
                next = Some(df);
                break;
            }

            while let Some(df) = next.take() {
                // Try to decode the next non-empty morsel first, so we know
                // whether the df is the last morsel.
                loop {
                    let Some(decode_fut) = decode_recv.recv().await else {
                        break;
                    };
                    let next_df = decode_fut.await?;
                    if next_df.height() == 0 {
                        continue;
                    }
                    next = Some(next_df);
                    break;
                }

                for df in split_to_morsels(
                    &df,
                    ideal_morsel_size,
                    next.is_none(),
                    last_morsel_min_split,
                ) {
                    if morsel_send
                        .send_morsel(Morsel::new(df, morsel_seq, source_token.clone()))
                        .await
                        .is_err()
                    {
                        return Ok(());
                    }
                    morsel_seq = morsel_seq.successor();
                }
            }
            PolarsResult::Ok(())
        });

        // Orchestration.
        let join_task = io_runtime.spawn(async move {
            prefetch_task.await.unwrap()?;
            decode_task.await.unwrap()?;
            distribute_task.await?;
            Ok(())
        });

        // Return.
        (morsel_recv, AbortOnDropHandle(join_task))
    }

    /// Creates a `RecordBatchDecoder` that turns `RecordBatchData` into DataFrames.
    /// Dictionaries must be loaded prior to initialization.
    pub(super) fn init_record_batch_decoder(&mut self) -> RecordBatchDecoder {
        dbg!("start init_record_batch_decoder"); //kdn

        debug_assert!(self.dictionaries.is_some());

        let file_metadata = self.metadata.clone();
        let projection_info = self.projection_info.clone();
        let dictionaries = self.dictionaries.clone();
        let row_index = self.row_index.clone();
        let slice_range = self.slice_range.clone();

        // kdn TODO
        // let max_morsel_size = get_max_morsel_size();

        RecordBatchDecoder {
            file_metadata,
            projection_info,
            dictionaries,
            row_index,
            slice_range,
            // max_morsel_size,
        }
    }
}
