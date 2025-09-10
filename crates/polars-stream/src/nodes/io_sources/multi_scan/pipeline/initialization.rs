use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use arrow::bitmap::Bitmap;
use futures::StreamExt;
use polars_core::prelude::PlHashMap;
use polars_error::PolarsResult;
use polars_io::predicates::ScanIOPredicate;
use polars_plan::plans::hive::HivePartitionsDf;
use polars_utils::slice_enum::Slice;

use crate::async_executor::{self, AbortOnDropHandle, TaskPriority};
use crate::async_primitives::connector::{self};
use crate::nodes::io_sources::multi_scan::components::bridge::{BridgeRecvPort, BridgeState};
use crate::nodes::io_sources::multi_scan::components::row_counter::RowCounter;
use crate::nodes::io_sources::multi_scan::components::row_deletions::{
    DeletionFilesProvider, ExternalFilterMask, RowDeletionsInit,
};
use crate::nodes::io_sources::multi_scan::config::MultiScanConfig;
use crate::nodes::io_sources::multi_scan::functions::resolve_slice::resolve_to_positive_slice;
use crate::nodes::io_sources::multi_scan::pipeline::models::{
    ExtraOperations, InitializedPipelineState, ResolvedSliceInfo, StartReaderArgsConstant,
};
use crate::nodes::io_sources::multi_scan::pipeline::tasks::attach_reader_to_bridge::AttachReaderToBridge;
use crate::nodes::io_sources::multi_scan::pipeline::tasks::bridge::spawn_bridge;
use crate::nodes::io_sources::multi_scan::pipeline::tasks::reader_starter::{
    InitializedReaderState, ReaderStarter,
};
use crate::nodes::io_sources::multi_scan::reader_interface::FileReader;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;

pub fn initialize_multi_scan_pipeline(config: Arc<MultiScanConfig>) -> InitializedPipelineState {
    assert!(config.num_pipelines() > 0);

    if config.verbose {
        eprintln!(
            "[MultiScanTaskInit]: \
            {} sources, \
            reader name: {}, \
            {:?}, \
            n_readers_pre_init: {}, \
            max_concurrent_scans: {}",
            config.sources.len(),
            config.file_reader_builder.reader_name(),
            config.reader_capabilities(),
            config.n_readers_pre_init(),
            config.max_concurrent_scans(),
        );
    }

    let bridge_state = Arc::new(Mutex::new(BridgeState::NotYetStarted));

    let (bridge_handle, bridge_recv_port_tx, phase_channel_tx) = spawn_bridge(bridge_state.clone());

    let task_handle =
        AbortOnDropHandle::new(async_executor::spawn(TaskPriority::Low, async move {
            finish_initialize_multi_scan_pipeline(config, bridge_recv_port_tx).await?;
            bridge_handle.await;
            Ok(())
        }));

    InitializedPipelineState {
        task_handle,
        phase_channel_tx,
        bridge_state,
    }
}

async fn finish_initialize_multi_scan_pipeline(
    config: Arc<MultiScanConfig>,
    bridge_recv_port_tx: connector::Sender<BridgeRecvPort>,
) -> PolarsResult<()> {
    let verbose = config.verbose;

    let (skip_files_mask, predicate) = initialize_predicate(
        config.predicate.as_ref(),
        config.hive_parts.as_deref(),
        verbose,
    )?;

    if let Some(skip_files_mask) = &skip_files_mask {
        assert_eq!(skip_files_mask.len(), config.sources.len());
    }

    if verbose {
        eprintln!(
            "[MultiScanTaskInit]: \
            predicate: {:?}, \
            skip files mask: {:?}, \
            predicate to reader: {:?}",
            config.predicate.is_some().then_some("<predicate>"),
            skip_files_mask.is_some().then_some("<skip_files>"),
            predicate.is_some().then_some("<predicate>"),
        )
    }

    #[expect(clippy::never_loop)]
    loop {
        if skip_files_mask
            .as_ref()
            .is_some_and(|x| x.unset_bits() == 0)
        {
            if verbose {
                eprintln!("[MultiScanTaskInit]: early return (skip_files_mask / predicate)")
            }
        } else if config.pre_slice.as_ref().is_some_and(|x| x.len() == 0) {
            if cfg!(debug_assertions) {
                panic!("should quit earlier");
            }

            if verbose {
                eprintln!("[MultiScanTaskInit]: early return (pre_slice.len == 0)")
            }
        } else {
            break;
        }

        return Ok(());
    }

    let predicate = predicate.cloned();

    let num_pipelines = config.num_pipelines();
    let reader_capabilities = config.reader_capabilities();

    if config.sources.first().is_some_and(|x| x.run_async())
        && reader_capabilities.contains(ReaderCapabilities::NEEDS_FILE_CACHE_INIT)
    {
        // In cloud execution the entries may not exist at this point due to DSL resolution
        // happening on a separate machine.
        polars_io::file_cache::init_entries_from_uri_list(
            config
                .sources
                .as_paths()
                .unwrap()
                .iter()
                .map(|path| Arc::from(path.to_str())),
            config.cloud_options.as_deref(),
        )?;
    }

    // Row index should only be pushed if we have a predicate or negative slice as there is a
    // serial synchronization cost from needing to track the row position.
    if config.row_index.is_some() {
        debug_assert!(
            config.predicate.is_some() || matches!(config.pre_slice, Some(Slice::Negative { .. }))
        );
    }

    let ResolvedSliceInfo {
        scan_source_idx,
        row_index,
        pre_slice,
        initialized_readers,
        row_deletions,
    } = match config.pre_slice {
        // This can hugely benefit NDJSON, as it can read backwards.
        Some(Slice::Negative { .. })
            if config.sources.len() == 1
                && reader_capabilities.contains(ReaderCapabilities::NEGATIVE_PRE_SLICE)
                && (config.row_index.is_none()
                    || reader_capabilities.contains(ReaderCapabilities::ROW_INDEX))
                && (config.deletion_files.is_none()
                    || reader_capabilities.contains(ReaderCapabilities::EXTERNAL_FILTER_MASK)) =>
        {
            if verbose {
                eprintln!("[MultiScanTaskInit]: Single file negative slice");
            }

            ResolvedSliceInfo {
                scan_source_idx: 0,
                row_index: config.row_index.clone(),
                pre_slice: config.pre_slice.clone(),
                initialized_readers: None,
                row_deletions: Default::default(),
            }
        },
        _ => {
            if let Some(Slice::Negative { .. }) = config.pre_slice {
                if verbose {
                    eprintln!("[MultiScanTaskInit]: Begin resolving negative slice to positive");
                }
            }

            resolve_to_positive_slice(&config).await?
        },
    };

    let initialized_row_deletions: Arc<PlHashMap<usize, ExternalFilterMask>> =
        Arc::new(row_deletions);

    let cast_columns_policy = config.cast_columns_policy.clone();
    let missing_columns_policy = config.missing_columns_policy;
    let include_file_paths = config.include_file_paths.clone();

    let extra_ops = ExtraOperations {
        row_index,
        row_index_col_idx: config.row_index.as_ref().map_or(usize::MAX, |x| {
            config.final_output_schema.index_of(&x.name).unwrap()
        }),
        pre_slice,
        include_file_paths,
        file_path_col_idx: config.include_file_paths.as_ref().map_or(usize::MAX, |x| {
            config.final_output_schema.index_of(x).unwrap()
        }),
        predicate,
    };

    if verbose {
        eprintln!(
            "[MultiScanTaskInit]: \
            scan_source_idx: {}, \
            extra_ops: {:?}",
            scan_source_idx, &extra_ops,
        )
    }

    // Pre-initialized readers if we resolved a negative slice.
    let mut initialized_readers: VecDeque<(Box<dyn FileReader>, RowCounter)> = initialized_readers
        .map(|(idx, readers)| {
            // Sanity check
            assert_eq!(idx, scan_source_idx);
            readers
        })
        .unwrap_or_default();

    let has_row_index_or_slice = extra_ops.has_row_index_or_slice();

    let config = config.clone();

    // Buffered initialization stream. This concurrently calls `FileReader::initialize()`,
    // allowing for e.g. concurrent Parquet metadata fetch.
    let readers_init_iter = {
        let skip_files_mask = skip_files_mask.clone();

        let mut range = {
            // If a negative slice was initialized, the length of the initialized readers will be the exact
            // stopping position.
            let end = if initialized_readers.is_empty() {
                config.sources.len()
            } else {
                scan_source_idx + initialized_readers.len()
            };

            scan_source_idx..end
        };

        if verbose {
            let n_filtered = skip_files_mask
                .clone()
                .map_or(0, |x| x.sliced(range.start, range.len()).set_bits());
            let n_readers_init = range.len() - n_filtered;

            eprintln!(
                "\
                [MultiScanTaskInit]: Readers init: {} / ({} total) \
                (range: {:?}, filtered out: {})",
                n_readers_init,
                config.sources.len(),
                &range,
                n_filtered,
            )
        }

        if let Some(skip_files_mask) = &skip_files_mask {
            range.end = range
                .end
                .min(skip_files_mask.len() - skip_files_mask.trailing_ones());
        }

        let range = range.filter(move |scan_source_idx| {
            let can_skip = !has_row_index_or_slice
                && skip_files_mask
                    .as_ref()
                    .is_some_and(|x| x.get_bit(*scan_source_idx));

            !can_skip
        });

        let sources = config.sources.clone();
        let cloud_options = config.cloud_options.clone();
        let file_reader_builder = config.file_reader_builder.clone();
        let deletion_files_provider = DeletionFilesProvider::new(config.deletion_files.clone());

        futures::stream::iter(range)
            .map(move |scan_source_idx| {
                let sources = sources.clone();
                let cloud_options = cloud_options.clone();
                let file_reader_builder = file_reader_builder.clone();
                let deletion_files_provider = deletion_files_provider.clone();
                let initialized_row_deletions = initialized_row_deletions.clone();

                let maybe_initialized = initialized_readers.pop_front();
                let scan_source = sources.get(scan_source_idx).unwrap().into_owned();

                AbortOnDropHandle::new(async_executor::spawn(TaskPriority::Low, async move {
                    let (scan_source, reader, n_rows_in_file) = async {
                        if verbose {
                            eprintln!("[MultiScan]: Initialize source {scan_source_idx}");
                        }

                        let scan_source = scan_source?;

                        if let Some((reader, n_rows_in_file)) = maybe_initialized {
                            return PolarsResult::Ok((scan_source, reader, Some(n_rows_in_file)));
                        }

                        let mut reader = file_reader_builder.build_file_reader(
                            scan_source.clone(),
                            cloud_options.clone(),
                            scan_source_idx,
                        );

                        reader.initialize().await?;
                        let opt_n_rows = reader
                            .fast_n_rows_in_file()
                            .await?
                            .map(|num_phys_rows| RowCounter::new(num_phys_rows, 0));

                        PolarsResult::Ok((scan_source, reader, opt_n_rows))
                    }
                    .await?;

                    let row_deletions: Option<RowDeletionsInit> = initialized_row_deletions
                        .get(&scan_source_idx)
                        .map(|x| RowDeletionsInit::Initialized(x.clone()))
                        .or_else(|| {
                            deletion_files_provider.spawn_row_deletions_init(
                                scan_source_idx,
                                cloud_options,
                                num_pipelines,
                                verbose,
                            )
                        });

                    Ok(InitializedReaderState {
                        scan_source_idx,
                        scan_source,
                        reader,
                        n_rows_in_file,
                        row_deletions,
                    })
                }))
            })
            .buffered(config.n_readers_pre_init().min(config.sources.len()))
    };

    let sources = config.sources.clone();
    let readers_init_iter = readers_init_iter.boxed();
    let hive_parts = config.hive_parts.clone();
    let final_output_schema = config.final_output_schema.clone();
    let file_projection_builder = config.file_projection_builder.clone();
    let max_concurrent_scans = config.max_concurrent_scans();

    let (started_reader_tx, started_reader_rx) =
        tokio::sync::mpsc::channel(max_concurrent_scans.max(2) - 1);

    let reader_starter_handle = AbortOnDropHandle::new(async_executor::spawn(
        TaskPriority::Low,
        ReaderStarter {
            reader_capabilities,
            n_sources: sources.len(),

            readers_init_iter,
            started_reader_tx,
            max_concurrent_scans,
            skip_files_mask,
            extra_ops,
            constant_args: StartReaderArgsConstant {
                hive_parts,
                final_output_schema,
                reader_capabilities,
                file_projection_builder,
                cast_columns_policy,
                missing_columns_policy,
                forbid_extra_columns: config.forbid_extra_columns.clone(),
                num_pipelines,
                verbose,
            },
            verbose,
        }
        .run(),
    ));

    let attach_to_bridge_handle = AbortOnDropHandle::new(async_executor::spawn(
        TaskPriority::Low,
        AttachReaderToBridge {
            started_reader_rx,
            bridge_recv_port_tx,
            verbose,
        }
        .run(),
    ));

    attach_to_bridge_handle.await?;
    reader_starter_handle.await?;

    Ok(())
}

/// # Returns
/// (skip_files_mask, predicate)
fn initialize_predicate<'a>(
    predicate: Option<&'a ScanIOPredicate>,
    hive_parts: Option<&HivePartitionsDf>,
    verbose: bool,
) -> PolarsResult<(Option<Bitmap>, Option<&'a ScanIOPredicate>)> {
    if let Some(predicate) = predicate {
        if let Some(hive_parts) = hive_parts {
            let mut skip_files_mask = None;

            if let Some(predicate) = &predicate.hive_predicate {
                let mask = predicate
                    .evaluate_io(hive_parts.df())?
                    .bool()?
                    .rechunk()
                    .into_owned()
                    .downcast_into_iter()
                    .next()
                    .unwrap()
                    .values()
                    .clone();

                // TODO: Optimize to avoid doing this
                let mask = !&mask;

                if verbose {
                    eprintln!(
                        "[MultiScan]: Predicate pushdown allows skipping {} / {} files",
                        mask.set_bits(),
                        mask.len()
                    );
                }

                skip_files_mask = Some(mask);
            }

            let need_pred_for_inner_readers = !predicate.hive_predicate_is_full_predicate;

            return Ok((
                skip_files_mask,
                need_pred_for_inner_readers.then_some(predicate),
            ));
        }
    }

    Ok((None, predicate))
}
