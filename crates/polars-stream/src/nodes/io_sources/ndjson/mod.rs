use std::cmp::Reverse;
use std::ops::Range;
use std::sync::Arc;

use chunk_reader::ChunkReader;
use line_batch_processor::{LineBatchProcessor, LineBatchProcessorOutputPort};
use negative_slice_pass::MorselStreamReverser;
use polars_core::config;
use polars_core::schema::SchemaRef;
use polars_core::utils::arrow::bitmap::Bitmap;
use polars_error::{PolarsResult, polars_bail, polars_err};
use polars_io::cloud::CloudOptions;
use polars_io::prelude::estimate_n_lines_in_file;
use polars_io::utils::compression::maybe_decompress_bytes;
use polars_io::{RowIndex, ndjson};
use polars_plan::dsl::{NDJsonReadOptions, ScanSource};
use polars_plan::plans::{FileInfo, ndjson_file_info};
use polars_plan::prelude::FileScanOptions;
use polars_utils::IdxSize;
use polars_utils::mem::prefetch::get_memory_prefetch_func;
use polars_utils::mmap::MemSlice;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::priority::Priority;
use row_index_limit_pass::ApplyRowIndexOrLimit;

use super::multi_scan::MultiScanable;
use super::{RowRestriction, SourceNode, SourceOutput};
use crate::async_executor::{AbortOnDropHandle, spawn};
use crate::async_primitives::connector::{Receiver, connector};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::SourceToken;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::io_sources::MorselOutput;
use crate::nodes::{MorselSeq, TaskPriority};
mod chunk_reader;
mod line_batch_distributor;
mod line_batch_processor;
mod negative_slice_pass;
mod row_index_limit_pass;

#[derive(Clone)]
pub struct NDJsonSourceNode {
    scan_source: ScanSource,
    file_info: FileInfo,
    file_options: Box<FileScanOptions>,
    options: NDJsonReadOptions,
    schema: Option<SchemaRef>,
    verbose: bool,
}

impl NDJsonSourceNode {
    pub fn new(
        scan_source: ScanSource,
        file_info: FileInfo,
        file_options: Box<FileScanOptions>,
        options: NDJsonReadOptions,
    ) -> Self {
        let verbose = config::verbose();

        Self {
            scan_source,
            file_info,
            file_options,
            options,
            schema: None,
            verbose,
        }
    }
}

impl SourceNode for NDJsonSourceNode {
    fn name(&self) -> &str {
        "ndjson_source"
    }

    fn is_source_output_parallel(&self, _is_receiver_serial: bool) -> bool {
        true
    }

    fn spawn_source(
        &mut self,
        mut output_recv: Receiver<SourceOutput>,
        state: &StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
        unrestricted_row_count: Option<tokio::sync::oneshot::Sender<IdxSize>>,
    ) {
        let verbose = self.verbose;

        self.schema = Some(self.file_info.reader_schema.take().unwrap().unwrap_right());

        let global_bytes = match self.scan_source_bytes() {
            Ok(v) => v,
            e @ Err(_) => {
                join_handles.push(spawn(TaskPriority::Low, async move {
                    e?;
                    unreachable!()
                }));
                return;
            },
        };

        let mut is_negative_slice = false;

        // Convert (offset, len) to Range
        // Note: This is converted to right-to-left for negative slice (i.e. range.start is position
        // from end).
        let global_slice: Option<Range<usize>> =
            if let Some((offset, len)) = self.file_options.pre_slice {
                if offset < 0 {
                    is_negative_slice = true;
                    // array: [_ _ _ _ _]
                    // slice: [    _ _  ]
                    // in:    offset: -3, len: 2
                    // out:   1..3 (right-to-left)
                    let offset_rev = -offset as usize;
                    Some(offset_rev.saturating_sub(len)..offset_rev)
                } else {
                    Some(offset as usize..offset as usize + len)
                }
            } else {
                None
            };

        let (total_row_count_tx, total_row_count_rx) =
            if is_negative_slice && self.file_options.row_index.is_some() {
                let (tx, rx) = tokio::sync::oneshot::channel();
                (Some(tx), Some(rx))
            } else {
                (None, None)
            };

        let needs_total_row_count =
            total_row_count_tx.is_some() || unrestricted_row_count.is_some();

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
                            "[NDJSON source]: n_lines_estimate: {}, line_length_estimate: {}",
                            n_lines_estimate, line_length_estimate
                        );
                    }

                    // Estimated stopping point in the file
                    x.end.saturating_mul(line_length_estimate)
                }
            } else {
                global_bytes.len()
            };

            let chunk_size = n_bytes_to_split.div_ceil(16 * state.num_pipelines);

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
                "[NDJSON source]: \
                chunk_size: {}, \
                n_chunks: {}, \
                row_index: {:?} \
                global_slice: {:?}, \
                is_negative_slice: {}",
                chunk_size,
                global_bytes.len().div_ceil(chunk_size),
                &self.file_options.row_index,
                &global_slice,
                is_negative_slice,
            );
        }

        let (mut phase_tx_senders, mut phase_tx_receivers) = (0..state.num_pipelines)
            .map(|_| connector::<MorselOutput>())
            .collect::<(Vec<_>, Vec<_>)>();

        // Note: This counts from the end of file for negative slice.
        let n_rows_to_skip = global_slice.as_ref().map_or(0, |x| x.start);

        let (opt_linearizer, mut linearizer_inserters) = if global_slice.is_some()
            || self.file_options.row_index.is_some()
        {
            let (a, b) =
                Linearizer::<Priority<Reverse<MorselSeq>, DataFrame>>::new(state.num_pipelines, 1);
            (Some(a), b)
        } else {
            (None, vec![])
        };

        let output_to_linearizer = opt_linearizer.is_some();

        let opt_post_process_handle = if is_negative_slice {
            // Note: This is right-to-left
            let negative_slice = global_slice.clone().unwrap();

            if verbose {
                eprintln!("[NDJSON source]: Initialize morsel stream reverser");
            }

            Some(AbortOnDropHandle::new(spawn(
                TaskPriority::High,
                MorselStreamReverser {
                    morsel_receiver: opt_linearizer.unwrap(),
                    phase_tx_receivers: std::mem::take(&mut phase_tx_receivers),
                    offset_len_rtl: (
                        negative_slice.start,
                        negative_slice.end - negative_slice.start,
                    ),
                    // The correct row index offset can only be known after total row count is
                    // available. This is handled by the MorselStreamReverser.
                    row_index: self
                        .file_options
                        .row_index
                        .take()
                        .map(|x| (x, total_row_count_rx.unwrap())),
                    verbose,
                }
                .run(),
            )))
        } else if global_slice.is_some() || self.file_options.row_index.is_some() {
            let mut row_index = self.file_options.row_index.take();

            if verbose {
                eprintln!("[NDJSON source]: Initialize ApplyRowIndexOrLimit");
            }

            if let Some(ri) = row_index.as_mut() {
                // Update the row index offset according to the slice start.
                let Some(v) = ri.offset.checked_add(n_rows_to_skip as IdxSize) else {
                    let offset = ri.offset;
                    join_handles.push(spawn(TaskPriority::Low, async move {
                        polars_bail!(
                            ComputeError:
                            "row_index with offset {} overflows at {} rows",
                            offset, n_rows_to_skip
                        );
                    }));
                    return;
                };
                ri.offset = v;
            }

            Some(AbortOnDropHandle::new(spawn(
                TaskPriority::High,
                ApplyRowIndexOrLimit {
                    morsel_receiver: opt_linearizer.unwrap(),
                    phase_tx_receivers: std::mem::take(&mut phase_tx_receivers),
                    // Note: The line batch distributor handles skipping lines until the offset,
                    // we only need to handle the limit here.
                    limit: global_slice.as_ref().map(|x| x.len()),
                    row_index,
                    verbose,
                }
                .run(),
            )))
        } else {
            None
        };

        let chunk_reader = match self.try_init_chunk_reader().map(Arc::new) {
            Ok(v) => v,
            e @ Err(_) => {
                join_handles.push(spawn(TaskPriority::Low, async move {
                    e?;
                    unreachable!()
                }));
                return;
            },
        };

        if !is_negative_slice {
            get_memory_prefetch_func(verbose)(global_bytes.as_ref());
        }

        let (line_batch_distribute_tx, line_batch_distribute_receivers) =
            distributor_channel(state.num_pipelines, 1);

        let source_token = SourceToken::new();
        // Initialize in reverse as we want to manually pop from either the linearizer or the phase receivers depending
        // on if we have negative slice.
        let line_batch_processor_handles = line_batch_distribute_receivers
            .into_iter()
            .enumerate()
            .rev()
            .map(|(worker_idx, line_batch_rx)| {
                let global_bytes = global_bytes.clone();
                let chunk_reader = chunk_reader.clone();
                let source_token = source_token.clone();

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
                                phase_tx: None,
                                phase_tx_receiver: phase_tx_receivers.pop().unwrap(),
                                source_token: source_token.clone(),
                                wait_group: WaitGroup::default(),
                            }
                        },
                        needs_total_row_count,

                        // Only log from the last worker to prevent flooding output.
                        verbose: verbose && worker_idx == state.num_pipelines - 1,
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

        join_handles.push(spawn(TaskPriority::Low, async move {
            let mut row_count = line_batch_distributor_task_handle.await?;

            if verbose {
                eprintln!("[NDJSON source]: line batch distributor handle returned");
            }

            for handle in line_batch_processor_handles {
                let n_rows_processed = handle.await?;
                if needs_total_row_count {
                    row_count = row_count.checked_add(n_rows_processed).unwrap();
                }
            }

            if verbose {
                eprintln!("[NDJSON source]: line batch processor handles returned");
            }

            if let Some(tx) = total_row_count_tx {
                assert!(needs_total_row_count);

                if verbose {
                    eprintln!(
                        "[NDJSON source]: \
                        send total row count: {}",
                        row_count
                    )
                }
                _ = tx.send(row_count);
            }

            if let Some(unrestricted_row_count) = unrestricted_row_count {
                assert!(needs_total_row_count);

                if verbose {
                    eprintln!(
                        "[NDJSON source]: send unrestricted_row_count: {}",
                        row_count
                    );
                }

                let num_rows = row_count;
                let num_rows = IdxSize::try_from(num_rows)
                    .map_err(|_| polars_err!(bigidx, ctx = "ndjson file", size = num_rows))?;
                _ = unrestricted_row_count.send(num_rows);
            }

            if let Some(handle) = opt_post_process_handle {
                handle.await?;
            }

            if verbose {
                eprintln!("[NDJSON source]: returning");
            }

            Ok(())
        }));

        join_handles.push(spawn(TaskPriority::Low, async move {
            // Every phase we are given a new send port.
            while let Ok(phase_output) = output_recv.recv().await {
                let source_token = SourceToken::new();
                let morsel_senders = phase_output.port.parallel();

                let mut morsel_outcomes = Vec::with_capacity(morsel_senders.len());

                for (phase_tx_senders, port) in phase_tx_senders.iter_mut().zip(morsel_senders) {
                    let (outcome, wait_group, morsel_output) =
                        MorselOutput::from_port(port, source_token.clone());
                    _ = phase_tx_senders.send(morsel_output).await;
                    morsel_outcomes.push((outcome, wait_group));
                }

                let mut is_finished = true;

                for (outcome, wait_group) in morsel_outcomes.into_iter() {
                    wait_group.wait().await;
                    is_finished &= outcome.did_finish();
                }

                if is_finished {
                    break;
                }

                phase_output.outcome.stop();
            }

            Ok(())
        }));
    }
}

impl NDJsonSourceNode {
    fn try_init_chunk_reader(&mut self) -> PolarsResult<ChunkReader> {
        ChunkReader::try_new(
            &self.options,
            self.schema.as_ref().unwrap(),
            self.file_options.with_columns.as_deref(),
        )
    }

    fn scan_source_bytes(&self) -> PolarsResult<MemSlice> {
        let run_async = self.scan_source.run_async();
        let source = self
            .scan_source
            .as_scan_source_ref()
            .to_memslice_async_assume_latest(run_async)?;

        let mem_slice = {
            let mut out = vec![];
            maybe_decompress_bytes(&source, &mut out)?;

            if out.is_empty() {
                source
            } else {
                MemSlice::from_vec(out)
            }
        };

        Ok(mem_slice)
    }
}

impl MultiScanable for NDJsonSourceNode {
    type ReadOptions = NDJsonReadOptions;

    const BASE_NAME: &'static str = "ndjson";
    const SPECIALIZED_PRED_PD: bool = false;

    async fn new(
        source: ScanSource,
        options: &Self::ReadOptions,
        cloud_options: Option<&CloudOptions>,
        row_index: Option<PlSmallStr>,
    ) -> PolarsResult<Self> {
        let has_row_index = row_index.as_ref().is_some();

        let file_options = Box::new(FileScanOptions {
            row_index: row_index.map(|name| RowIndex { name, offset: 0 }),
            ..Default::default()
        });

        let ndjson_options = options.clone();
        let mut file_info = ndjson_file_info(
            &source.clone().into_sources(),
            &file_options,
            &ndjson_options,
            cloud_options,
        )?;

        let schema = Arc::make_mut(&mut file_info.schema);

        if has_row_index {
            // @HACK: This is really hacky because the NDJSON schema wrongfully adds the row index.
            schema.shift_remove(
                file_options
                    .row_index
                    .as_ref()
                    .map(|x| x.name.as_str())
                    .unwrap(),
            );
        }

        for (name, dtype) in schema.iter_mut() {
            if let Some(dtype_override) = options
                .schema_overwrite
                .as_ref()
                .and_then(|x| x.get(name))
                .or_else(|| options.schema.as_ref().and_then(|x| x.get(name)))
            {
                *dtype = dtype_override.clone();
            }
        }

        Ok(Self::new(source, file_info, file_options, ndjson_options))
    }

    fn with_projection(&mut self, projection: Option<&Bitmap>) {
        self.file_options.with_columns = projection.map(|p| {
            p.true_idx_iter()
                .map(|idx| self.file_info.schema.get_at_index(idx).unwrap().0.clone())
                .collect()
        });
    }

    fn with_row_restriction(&mut self, row_restriction: Option<RowRestriction>) {
        self.file_options.pre_slice = None;

        match row_restriction {
            None => {},
            Some(RowRestriction::Slice(rng)) => {
                self.file_options.pre_slice = Some((rng.start as i64, rng.end - rng.start))
            },
            Some(RowRestriction::Predicate(_)) => unreachable!(),
        }
    }

    async fn unrestricted_row_count(&mut self) -> PolarsResult<IdxSize> {
        let mem_slice = self.scan_source_bytes()?;

        // TODO: Parallelize this over the async executor
        let num_rows = ndjson::count_rows(&mem_slice);
        let num_rows = IdxSize::try_from(num_rows)
            .map_err(|_| polars_err!(bigidx, ctx = "ndjson file", size = num_rows))?;
        Ok(num_rows)
    }

    async fn physical_schema(&mut self) -> PolarsResult<SchemaRef> {
        Ok(self.file_info.schema.clone())
    }
}
