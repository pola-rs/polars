use arrow::bitmap::Bitmap;
use components::bridge::BridgeRecvPort;
use components::row_deletions::{DeletionFilesProvider, ExternalFilterMask, RowDeletionsInit};
use components::{ExtraOperations, ForbidExtraColumns};
use futures::StreamExt;
use futures::stream::BoxStream;
use polars_error::PolarsResult;
use polars_plan::dsl::ScanSource;
use polars_utils::IdxSize;
use polars_utils::slice_enum::Slice;

use crate::async_executor::{self, AbortOnDropHandle, TaskPriority};
use crate::async_primitives::connector;
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::nodes::io_sources::multi_file_reader::components;
use crate::nodes::io_sources::multi_file_reader::functions::resolve_slice::{
    ResolvedSliceInfo, resolve_to_positive_slice,
};
use crate::nodes::io_sources::multi_file_reader::reader_interface::capabilities::ReaderCapabilities;
use crate::nodes::io_sources::multi_file_reader::reader_interface::{
    BeginReadArgs, FileReader, FileReaderCallbacks, Projection,
};

/// Starts readers, potentially multiple at the same time if it can.
struct ReaderStarter {
    reader_capabilities: ReaderCapabilities,
    readers_init_iter: BoxStream<'static, PolarsResult<InitializedReaderState>>,
    n_sources: usize,
    started_reader_tx: tokio::sync::mpsc::Sender<(
        AbortOnDropHandle<PolarsResult<StartedReaderState>>,
        WaitToken,
    )>,
    max_concurrent_scans: usize,
    skip_files_mask: Option<Bitmap>,
    extra_ops: ExtraOperations,
    constant_args: StartReaderArgsConstant,
    verbose: bool,
}

pub struct InitializedReaderState {
    scan_source_idx: usize,
    scan_source: ScanSource,
    reader: Box<dyn FileReader>,
    n_rows_in_file: Option<RowCounter>,
    row_deletions: Option<RowDeletionsInit>,
}

impl ReaderStarter {
    async fn run(self) -> PolarsResult<()> {
        let ReaderStarter {
            reader_capabilities,
            mut readers_init_iter,
            n_sources,
            started_reader_tx,
            max_concurrent_scans,
            skip_files_mask,
            extra_ops,
            constant_args,
            verbose,
        } = self;

        // Notes:
        // * This is unused if we aren't slicing or row indexing.
        let mut current_row_position: Option<RowCounter> = extra_ops
            .has_row_index_or_slice()
            .then_some(RowCounter::default());

        let wait_group = WaitGroup::default();

        loop {
            // Note: This loop should only do basic bookkeeping (e.g. slice position) and reader initialization.
            // It should avoid doing compute as much as possible - those should instead be deferred to spawned tasks.

            let pre_slice_this_file = extra_ops
                .pre_slice
                .clone()
                .map(|x| {
                    PolarsResult::Ok(match x {
                        Slice::Positive { .. } => {
                            x.offsetted(current_row_position.unwrap().num_rows()?)
                        },
                        Slice::Negative { .. } => x,
                    })
                })
                .transpose()?;

            if current_row_position.is_some() && verbose {
                eprintln!(
                    "[ReaderStarter]: \
                    current_row_position: {current_row_position:?}, \
                    pre_slice_this_file: {pre_slice_this_file:?}"
                )
            }

            if pre_slice_this_file.as_ref().is_some_and(|x| x.len() == 0) {
                if verbose {
                    eprintln!("[ReaderStarter]: Stopping (pre_slice)")
                }
                break;
            }

            let Some(InitializedReaderState {
                scan_source_idx,
                scan_source,
                mut reader,
                mut n_rows_in_file,
                row_deletions,
            }) = readers_init_iter.next().await.transpose()?
            else {
                if verbose {
                    eprintln!("[ReaderStarter]: Stopping (no more readers)")
                }
                break;
            };

            if verbose {
                eprintln!("[ReaderStarter]: scan_source_idx: {scan_source_idx}")
            }

            // Note: We `.await` here for the row deletions to be fully loaded.
            //       For this reason it's important that we already spawn background tasks to fully
            //       load them at the reader pre-initialization stage.
            let external_filter_mask = if let Some(row_deletions) = row_deletions {
                Some(row_deletions.into_external_filter_mask().await?)
            } else {
                None
            };

            // * This translates the `pre_slice` to physical offsets (i.e. apply before deleting rows).
            //   The slice remains the same if there are no row deletions.
            let pre_slice_this_file: Option<PhysicalSlice> =
                pre_slice_this_file.map(|pre_slice| match pre_slice {
                    Slice::Positive { .. } => {
                        PhysicalSlice::new(pre_slice, external_filter_mask.as_ref())
                    },

                    // This is hit here for NDJSON single file negative slice, we just passthrough
                    // in this case.
                    Slice::Negative { .. } => {
                        if external_filter_mask.is_some() {
                            unimplemented!(
                                "{pre_slice:?} {}",
                                ExternalFilterMask::log_display(external_filter_mask.as_ref())
                            )
                        }

                        assert!(
                            (extra_ops.row_index.is_none()
                                || reader_capabilities.contains(ReaderCapabilities::ROW_INDEX))
                                && (external_filter_mask.is_none()
                                    || reader_capabilities
                                        .contains(ReaderCapabilities::EXTERNAL_FILTER_MASK))
                        );

                        PhysicalSlice {
                            slice: pre_slice,
                            // Note, this is not the correct starting position. The assertion above
                            // should ensure this value is not used in post-apply.
                            slice_start_position: RowCounter::default(),
                        }
                    },
                });

            let row_index_this_file = {
                let current_row_position = if let Some(current_row_position) = current_row_position
                {
                    current_row_position.num_rows_idxsize_saturating()?
                } else {
                    IdxSize::MAX
                };

                extra_ops.row_index.clone().map(|mut ri| {
                    ri.offset = ri.offset.saturating_add(current_row_position);
                    ri
                })
            };

            let extra_ops_this_file = ExtraOperations {
                row_index: row_index_this_file,
                pre_slice: pre_slice_this_file
                    .as_ref()
                    .map(|phys_slice: &PhysicalSlice| phys_slice.slice.clone()),
                // Other operations don't need updating per file
                ..extra_ops.clone()
            };

            // &str that holds the reason
            let mut skip_read_reason: Option<&'static str> = skip_files_mask
                .as_ref()
                .is_some_and(|x| x.get_bit(scan_source_idx))
                .then_some("skip_files_mask");

            if skip_read_reason.is_some() {
                // If this is not the case then the reader does not need to be sent here.
                debug_assert!(extra_ops.has_row_index_or_slice())
            }

            // `fast_n_rows_in_file()` or negative slice, we know the exact row count here already.
            // After this point, if n_rows_in_file is `Some`, it should contain the exact physical
            // and deleted row counts.
            if let Some(n_rows_in_file) = n_rows_in_file.as_mut() {
                if let Some(external_filter_mask) = external_filter_mask.as_ref() {
                    unsafe {
                        n_rows_in_file.set_deleted_rows(external_filter_mask.num_deleted_rows())
                    }
                }

                if n_rows_in_file.num_rows()? == 0 {
                    skip_read_reason = Some("0 rows")
                } else if pre_slice_this_file.as_ref().is_some_and(|phys_slice| {
                    phys_slice
                        .slice
                        .clone()
                        .restrict_to_bounds(n_rows_in_file.num_physical_rows())
                        .len()
                        == 0
                }) {
                    skip_read_reason = Some("0 rows after slice")
                }
            }

            if let Some(skip_read_reason) = skip_read_reason {
                if verbose {
                    eprintln!(
                        "[ReaderStarter]: scan_source_idx: {scan_source_idx}: \
                        skip read ({skip_read_reason}): \
                        n_rows_in_file: {n_rows_in_file:?}, \
                        pre_slice: {pre_slice_this_file:?}"
                    )
                }

                if started_reader_tx.is_closed() {
                    break;
                }

                // We are tracking the row position so we need the row count from this file even if it's skipped.
                if extra_ops.has_row_index_or_slice() {
                    let Some(current_row_position) = current_row_position.as_mut() else {
                        panic!()
                    };

                    let pre_slice_this_file =
                        pre_slice_this_file.map(|phys_slice| phys_slice.slice);

                    // Should never: Negative slice should only hit this loop in the case:
                    // * Single NDJSON file that is not filtered out.
                    if let Some(Slice::Negative { .. }) = pre_slice_this_file {
                        panic!();
                    }

                    let get_row_count = async {
                        let num_physical_rows =
                            reader.row_position_after_slice(pre_slice_this_file).await?;

                        let num_deleted_rows = external_filter_mask.as_ref().map_or(0, |mask| {
                            mask.slice(
                                0,
                                mask.len().min(usize::try_from(num_physical_rows).unwrap()),
                            )
                            .num_deleted_rows()
                        });

                        let file_row_count = RowCounter::new(num_physical_rows, num_deleted_rows);

                        if verbose {
                            eprintln!(
                                "[ReaderStarter]: scan_source_idx: {scan_source_idx}: \
                                file_row_count: {file_row_count:?}"
                            )
                        }

                        PolarsResult::Ok(file_row_count)
                    };

                    if n_rows_in_file.is_none() {
                        n_rows_in_file = Some(get_row_count.await?)
                    } else if cfg!(debug_assertions) {
                        assert_eq!(n_rows_in_file.unwrap(), get_row_count.await?)
                    }

                    *current_row_position = current_row_position.add(n_rows_in_file.unwrap());
                }

                continue;
            }

            let (row_position_on_end_tx, row_position_on_end_rx) = if n_rows_in_file.is_none()
                && extra_ops.has_row_index_or_slice()
                && n_sources - scan_source_idx > 1
            {
                let (tx, rx) = connector::connector();
                (Some(tx), Some(rx))
            } else {
                (None, None)
            };

            let callbacks = FileReaderCallbacks {
                row_position_on_end_tx,
                ..Default::default()
            };

            let start_args_this_file = StartReaderArgsPerFile {
                scan_source,
                scan_source_idx,
                reader,
                pre_slice_this_file,
                extra_ops_this_file,
                callbacks,
                external_filter_mask: external_filter_mask.clone(),
            };

            let reader_start_task_handle = AbortOnDropHandle::new(async_executor::spawn(
                TaskPriority::Low,
                start_reader_impl(constant_args.clone(), start_args_this_file),
            ));

            if started_reader_tx
                .send((reader_start_task_handle, wait_group.token()))
                .await
                .is_err()
            {
                break;
            };

            // If we have row index or slice, we must wait for the row position callback before
            // we can start the next reader. This will be very fast for e.g. Parquet / IPC, but
            // for CSV / NDJSON this will be slower.
            //
            // Note: If this reader ends early due to an error, we may start the next reader with an incorrect
            // row position. But downstream will never connect the next reader to the bridge as it should join
            // on this reader and already exit from the error.
            //
            // TODO:
            // * Parallelize the CSV row count
            // * NDJSON skips rows (i.e. non-zero offset) in a single-threaded manner.
            if let Some(current_row_position) = current_row_position.as_mut() {
                let mut row_position_this_file = RowCounter::default();

                #[expect(clippy::never_loop)]
                loop {
                    if let Some(v) = n_rows_in_file {
                        row_position_this_file = v;
                        break;
                    };

                    // Note, can be None on the last scan source.
                    let Some(mut rx) = row_position_on_end_rx else {
                        break;
                    };

                    let Ok(num_physical_rows) = rx.recv().await else {
                        break;
                    };

                    let num_deleted_rows = external_filter_mask.map_or(0, |external_filter_mask| {
                        external_filter_mask
                            .slice(0, usize::try_from(num_physical_rows).unwrap())
                            .num_deleted_rows()
                    });

                    row_position_this_file = RowCounter::new(num_physical_rows, num_deleted_rows);
                    break;
                }

                *current_row_position = current_row_position.add(row_position_this_file);
            }

            if skip_read_reason.is_none() && max_concurrent_scans == 1 {
                if verbose {
                    eprintln!("[ReaderStarter]: max_concurrent_scans is 1, waiting..")
                }

                wait_group.wait().await;
            }
        }

        Ok(())
    }
}
