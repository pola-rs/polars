pub mod builder;

use std::cmp::Reverse;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use chunk_reader::ChunkReader;
use line_batch_processor::{LineBatchProcessor, LineBatchProcessorOutputPort};
use negative_slice_pass::MorselStreamReverser;
use polars_core::schema::SchemaRef;
use polars_error::{PolarsResult, polars_bail, polars_err};
use polars_io::cloud::CloudOptions;
use polars_io::prelude::estimate_n_lines_in_file;
use polars_io::utils::compression::maybe_decompress_bytes;
use polars_plan::dsl::{NDJsonReadOptions, ScanSource};
use polars_utils::IdxSize;
use polars_utils::mem::prefetch::get_memory_prefetch_func;
use polars_utils::mmap::MemSlice;
use polars_utils::priority::Priority;
use polars_utils::slice_enum::Slice;
use row_index_limit_pass::ApplyRowIndexOrLimit;

use super::multi_file_reader::reader_interface::output::FileReaderOutputRecv;
use super::multi_file_reader::reader_interface::{BeginReadArgs, FileReader, FileReaderCallbacks};
use crate::async_executor::{AbortOnDropHandle, spawn};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::morsel::SourceToken;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::io_sources::multi_file_reader::reader_interface::output::FileReaderOutputSend;
use crate::nodes::{MorselSeq, TaskPriority};
mod chunk_reader;
mod line_batch_distributor;
mod line_batch_processor;
mod negative_slice_pass;
mod row_index_limit_pass;

#[derive(Clone)]
pub struct NDJsonFileReader {
    scan_source: ScanSource,
    #[expect(unused)] // Will be used when implementing cloud streaming.
    cloud_options: Option<Arc<CloudOptions>>,
    options: Arc<NDJsonReadOptions>,
    verbose: bool,
    // Cached on first access - we may be called multiple times e.g. on negative slice.
    cached_bytes: Option<MemSlice>,
}

#[async_trait]
impl FileReader for NDJsonFileReader {
    async fn initialize(&mut self) -> PolarsResult<()> {
        Ok(())
    }

    fn begin_read(
        &mut self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)> {
        let verbose = self.verbose;

        let BeginReadArgs {
            projected_schema,
            mut row_index,
            pre_slice,

            num_pipelines,
            callbacks:
                FileReaderCallbacks {
                    file_schema_tx,
                    n_rows_in_file_tx,
                    row_position_on_end_tx,
                },

            predicate: None,
            cast_columns_policy: _,
        } = args
        else {
            panic!("unsupported args: {:?}", &args)
        };

        // TODO: This currently downloads and decompresses everything upfront in a blocking manner.
        // Ideally we have a streaming download/decompression.
        let global_bytes = self.get_bytes_maybe_decompress()?;

        // NDJSON: We just use the projected schema - the parser will automatically append NULL if
        // the field is not found.
        //
        // TODO
        // We currently always use the projected dtype, but this may cause
        // issues e.g. with temporal types. This can be improved to better choose
        // between the 2 dtypes.
        let schema = projected_schema;

        if let Some(mut tx) = file_schema_tx {
            _ = tx.try_send(schema.clone())
        }

        let is_negative_slice = matches!(pre_slice, Some(Slice::Negative { .. }));

        // Convert (offset, len) to Range
        // Note: This is converted to right-to-left for negative slice (i.e. range.start is position
        // from end).
        let global_slice: Option<Range<usize>> = if let Some(slice) = pre_slice.clone() {
            match slice {
                Slice::Positive { offset, len } => Some(offset..offset.saturating_add(len)),
                Slice::Negative {
                    offset_from_end,
                    len,
                } => {
                    // array: [_ _ _ _ _]
                    // slice: [    _ _  ]
                    // in:    offset_from_end: 3, len: 2
                    // out:   1..3 (right-to-left)
                    Some(offset_from_end.saturating_sub(len)..offset_from_end)
                },
            }
        } else {
            None
        };

        let (total_row_count_tx, total_row_count_rx) = if is_negative_slice && row_index.is_some() {
            let (tx, rx) = tokio::sync::oneshot::channel();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        let needs_total_row_count = total_row_count_tx.is_some()
            || n_rows_in_file_tx.is_some()
            || (row_position_on_end_tx.is_some()
                && matches!(pre_slice, Some(Slice::Negative { .. })));

        let chunk_size: usize = {
            let n_bytes_to_split = if let Some(x) = global_slice.as_ref() {
                if needs_total_row_count {
                    global_bytes.len()
                } else {
                    // There may be early stopping, try to heuristically use a smaller chunk size to stop faster.
                    let n_rows_to_sample = 8;
                    let n_lines_estimate =
                        estimate_n_lines_in_file(global_bytes.as_ref(), n_rows_to_sample);
                    let line_length_estimate = global_bytes.len().div_ceil(n_lines_estimate);

                    if verbose {
                        eprintln!(
                            "[NDJsonFileReader]: n_lines_estimate: {}, line_length_estimate: {}",
                            n_lines_estimate, line_length_estimate
                        );
                    }

                    // Estimated stopping point in the file
                    x.end.saturating_mul(line_length_estimate)
                }
            } else {
                global_bytes.len()
            };

            let chunk_size = n_bytes_to_split.div_ceil(16 * num_pipelines);

            let max_chunk_size = 16 * 1024 * 1024;
            // Use a small min chunk size to catch failures in tests.
            #[cfg(debug_assertions)]
            let min_chunk_size = 64;
            #[cfg(not(debug_assertions))]
            let min_chunk_size = 1024 * 4;

            let chunk_size = chunk_size.clamp(min_chunk_size, max_chunk_size);

            std::env::var("POLARS_FORCE_NDJSON_CHUNK_SIZE").map_or(chunk_size, |x| {
                x.parse::<usize>()
                    .expect("expected `POLARS_FORCE_NDJSON_CHUNK_SIZE` to be an integer")
            })
        };

        if verbose {
            eprintln!(
                "[NDJsonFileReader]: \
                project: {}, \
                global_slice: {:?}, \
                row_index: {:?}, \
                chunk_size: {}, \
                n_chunks: {}, \
                is_negative_slice: {}",
                schema.len(),
                &global_slice,
                &row_index,
                chunk_size,
                global_bytes.len().div_ceil(chunk_size),
                is_negative_slice,
            );
        }

        // Note: This counts from the end of file for negative slice.
        let n_rows_to_skip = global_slice.as_ref().map_or(0, |x| x.start);

        let (opt_linearizer, mut linearizer_inserters) =
            if global_slice.is_some() || row_index.is_some() {
                let (a, b) =
                    Linearizer::<Priority<Reverse<MorselSeq>, DataFrame>>::new(num_pipelines, 1);
                (Some(a), b)
            } else {
                (None, vec![])
            };

        let output_to_linearizer = opt_linearizer.is_some();

        let mut output_port = None;

        let opt_post_process_handle = if is_negative_slice {
            // Note: This is right-to-left
            let negative_slice = global_slice.clone().unwrap();

            if verbose {
                eprintln!("[NDJsonFileReader]: Initialize morsel stream reverser");
            }

            let (morsel_senders, rx) = FileReaderOutputSend::new_parallel(num_pipelines);
            output_port = Some(rx);

            Some(AbortOnDropHandle::new(spawn(
                TaskPriority::High,
                MorselStreamReverser {
                    morsel_receiver: opt_linearizer.unwrap(),
                    morsel_senders,
                    offset_len_rtl: (
                        negative_slice.start,
                        negative_slice.end - negative_slice.start,
                    ),
                    // The correct row index offset can only be known after total row count is
                    // available. This is handled by the MorselStreamReverser.
                    row_index: row_index.take().map(|x| (x, total_row_count_rx.unwrap())),
                    verbose,
                }
                .run(),
            )))
        } else if global_slice.is_some() || row_index.is_some() {
            let mut row_index = row_index.take();

            if verbose {
                eprintln!("[NDJsonFileReader]: Initialize ApplyRowIndexOrLimit");
            }

            if let Some(ri) = row_index.as_mut() {
                // Update the row index offset according to the slice start.
                let Some(v) = ri.offset.checked_add(n_rows_to_skip as IdxSize) else {
                    let offset = ri.offset;

                    polars_bail!(
                        ComputeError:
                        "row_index with offset {} overflows at {} rows",
                        offset, n_rows_to_skip
                    )
                };
                ri.offset = v;
            }

            let (morsel_tx, rx) = FileReaderOutputSend::new_serial();
            output_port = Some(rx);

            let limit = global_slice.as_ref().map(|x| x.len());

            let task = ApplyRowIndexOrLimit {
                morsel_receiver: opt_linearizer.unwrap(),
                morsel_tx,
                // Note: The line batch distributor handles skipping lines until the offset,
                // we only need to handle the limit here.
                limit,
                row_index,
                verbose,
            };

            if limit == Some(0) {
                None
            } else {
                Some(AbortOnDropHandle::new(spawn(
                    TaskPriority::High,
                    task.run(),
                )))
            }
        } else {
            None
        };

        let schema = Arc::new(schema);
        let chunk_reader = Arc::new(self.try_init_chunk_reader(&schema)?);

        if !is_negative_slice {
            get_memory_prefetch_func(verbose)(global_bytes.as_ref());
        }

        let (line_batch_distribute_tx, line_batch_distribute_receivers) =
            distributor_channel(num_pipelines, 1);

        let mut morsel_senders = if !output_to_linearizer {
            let (senders, outp) = FileReaderOutputSend::new_parallel(num_pipelines);
            assert!(output_port.is_none());
            output_port = Some(outp);
            senders
        } else {
            vec![]
        };

        // Initialize in reverse as we want to manually pop from either the linearizer or the phase receivers depending
        // on if we have negative slice.
        let line_batch_processor_handles = line_batch_distribute_receivers
            .into_iter()
            .enumerate()
            .rev()
            .map(|(worker_idx, line_batch_rx)| {
                let global_bytes = global_bytes.clone();
                let chunk_reader = chunk_reader.clone();
                // Note: We don't use this (it is handled by the bridge). But morsels require a source token.
                let source_token = SourceToken::new();

                AbortOnDropHandle::new(spawn(
                    TaskPriority::Low,
                    LineBatchProcessor {
                        worker_idx,

                        global_bytes,
                        chunk_reader,

                        line_batch_rx,
                        output_port: if output_to_linearizer {
                            LineBatchProcessorOutputPort::Linearize {
                                tx: linearizer_inserters.pop().unwrap(),
                            }
                        } else {
                            LineBatchProcessorOutputPort::Direct {
                                tx: morsel_senders.pop().unwrap(),
                                source_token: source_token.clone(),
                            }
                        },
                        needs_total_row_count,

                        // Only log from the last worker to prevent flooding output.
                        verbose: verbose && worker_idx == num_pipelines - 1,
                    }
                    .run(),
                ))
            })
            .collect::<Vec<_>>();

        let line_batch_distributor_task_handle = AbortOnDropHandle::new(spawn(
            TaskPriority::Low,
            line_batch_distributor::LineBatchDistributor {
                global_bytes,
                chunk_size,
                n_rows_to_skip,
                reverse: is_negative_slice,
                line_batch_distribute_tx,
            }
            .run(),
        ));

        let finishing_handle = spawn(TaskPriority::Low, async move {
            // Number of rows skipped by the line batch distributor.
            let n_rows_skipped: usize = line_batch_distributor_task_handle.await?;
            // Number of rows processed by the line batch processors.
            let mut n_rows_processed: usize = 0;

            if verbose {
                eprintln!("[NDJsonFileReader]: line batch distributor handle returned");
            }

            for handle in line_batch_processor_handles {
                n_rows_processed = n_rows_processed.saturating_add(handle.await?);
            }

            let total_row_count =
                needs_total_row_count.then_some(n_rows_skipped.saturating_add(n_rows_processed));

            if verbose {
                eprintln!("[NDJsonFileReader]: line batch processor handles returned");
            }

            if let Some(mut row_position_on_end_tx) = row_position_on_end_tx {
                let n = match pre_slice {
                    None => n_rows_skipped.saturating_add(n_rows_processed),

                    Some(Slice::Positive { offset, len }) => n_rows_skipped
                        .saturating_add(n_rows_processed)
                        .min(offset.saturating_add(len)),

                    Some(Slice::Negative { .. }) => {
                        total_row_count.unwrap().saturating_sub(n_rows_skipped)
                    },
                };

                let n = IdxSize::try_from(n)
                    .map_err(|_| polars_err!(bigidx, ctx = "ndjson file", size = n))?;

                _ = row_position_on_end_tx.try_send(n);
            }

            if let Some(tx) = total_row_count_tx {
                let total_row_count = total_row_count.unwrap();

                if verbose {
                    eprintln!(
                        "[NDJsonFileReader]: \
                        send total row count: {}",
                        total_row_count
                    )
                }
                _ = tx.send(total_row_count);
            }

            if let Some(mut n_rows_in_file_tx) = n_rows_in_file_tx {
                let total_row_count = total_row_count.unwrap();

                if verbose {
                    eprintln!(
                        "[NDJsonFileReader]: send n_rows_in_file: {}",
                        total_row_count
                    );
                }

                let num_rows = total_row_count;
                let num_rows = IdxSize::try_from(num_rows)
                    .map_err(|_| polars_err!(bigidx, ctx = "ndjson file", size = num_rows))?;
                _ = n_rows_in_file_tx.try_send(num_rows);
            }

            if let Some(handle) = opt_post_process_handle {
                handle.await?;
            }

            if verbose {
                eprintln!("[NDJsonFileReader]: returning");
            }

            Ok(())
        });

        Ok((output_port.unwrap(), finishing_handle))
    }
}

impl NDJsonFileReader {
    fn try_init_chunk_reader(&self, schema: &SchemaRef) -> PolarsResult<ChunkReader> {
        ChunkReader::try_new(&self.options, schema)
    }

    fn get_bytes_maybe_decompress(&mut self) -> PolarsResult<MemSlice> {
        if self.cached_bytes.is_none() {
            let run_async = self.scan_source.run_async();
            let source = self
                .scan_source
                .as_scan_source_ref()
                .to_memslice_async_assume_latest(run_async)?;

            let memslice = {
                let mut out = vec![];
                maybe_decompress_bytes(&source, &mut out)?;

                if out.is_empty() {
                    source
                } else {
                    MemSlice::from_vec(out)
                }
            };

            self.cached_bytes = Some(memslice);
        }

        Ok(self.cached_bytes.clone().unwrap())
    }
}
