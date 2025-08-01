use std::collections::VecDeque;

use components::row_deletions::DeletionFilesProvider;
use futures::StreamExt;
use polars_core::prelude::{InitHashMaps, PlHashMap};
use polars_error::PolarsResult;
use polars_utils::slice_enum::Slice;

use crate::async_executor::{self, AbortOnDropHandle, TaskPriority};
use crate::nodes::io_sources::multi_scan::components::row_counter::RowCounter;
use crate::nodes::io_sources::multi_scan::pipeline::models::ResolvedSliceInfo;
use crate::nodes::io_sources::multi_scan::{MultiScanConfig, components};

pub async fn resolve_to_positive_slice(
    config: &MultiScanConfig,
) -> PolarsResult<ResolvedSliceInfo> {
    match config.pre_slice.clone() {
        None => Ok(ResolvedSliceInfo {
            scan_source_idx: 0,
            row_index: config.row_index.clone(),
            pre_slice: None,
            initialized_readers: None,
            row_deletions: Default::default(),
        }),

        pre_slice @ Some(Slice::Positive { .. }) => Ok(ResolvedSliceInfo {
            scan_source_idx: 0,
            row_index: config.row_index.clone(),
            pre_slice,
            initialized_readers: None,
            row_deletions: Default::default(),
        }),

        Some(_) => resolve_negative_slice(config).await,
    }
}

/// Translates a negative slice to positive slice.
async fn resolve_negative_slice(config: &MultiScanConfig) -> PolarsResult<ResolvedSliceInfo> {
    let verbose = config.verbose;

    let pre_slice @ Slice::Negative {
        offset_from_end,
        len: slice_len,
    } = config.pre_slice.clone().unwrap()
    else {
        unreachable!()
    };

    if verbose {
        eprintln!(
            "resolve_negative_slice(): {:?}",
            Slice::Negative {
                offset_from_end,
                len: slice_len,
            }
        )
    }

    // Avoid traversal if we have no len.
    if slice_len == 0 {
        return Ok(ResolvedSliceInfo {
            scan_source_idx: config.sources.len(),
            row_index: config.row_index.clone(),
            pre_slice: Some(Slice::Positive { offset: 0, len: 0 }),
            initialized_readers: None,
            row_deletions: Default::default(),
        });
    }

    let deletion_files_provider = DeletionFilesProvider::new(config.deletion_files.clone());
    let num_pipelines = config.num_pipelines();

    let mut initialized_readers =
        VecDeque::with_capacity(config.sources.len().min(num_pipelines.saturating_add(4)));
    let mut file_row_deletions = PlHashMap::with_capacity(
        config
            .deletion_files
            .as_ref()
            .map_or(0, |x| x.num_files_with_deletions())
            .min(num_pipelines.saturating_add(4)),
    );

    let mut readers_init_iter = futures::stream::iter((0..config.sources.len()).rev())
        .map(|scan_source_idx| {
            let sources = config.sources.clone();
            let cloud_options = config.cloud_options.clone();
            let file_reader_builder = config.file_reader_builder.clone();
            let deletion_files_provider = deletion_files_provider.clone();

            AbortOnDropHandle::new(async_executor::spawn(TaskPriority::Low, async move {
                let mut reader =
                    sources
                        .get(scan_source_idx)
                        .unwrap()
                        .into_owned()
                        .map(|source| {
                            file_reader_builder.build_file_reader(
                                source,
                                cloud_options.clone(),
                                scan_source_idx,
                            )
                        })?;

                if verbose {
                    eprintln!("resolve_negative_slice(): init scan source {scan_source_idx}");
                }

                let row_deletions = deletion_files_provider.spawn_row_deletions_init(
                    scan_source_idx,
                    cloud_options,
                    num_pipelines,
                    verbose,
                );

                reader.initialize().await?;
                PolarsResult::Ok((scan_source_idx, reader, row_deletions))
            }))
        })
        .buffered(config.n_readers_pre_init());

    let n_rows_needed: usize = offset_from_end;

    let mut n_rows_seen: RowCounter = RowCounter::default();
    let mut n_rows_trimmed: RowCounter = RowCounter::default();
    let mut n_files_from_end: usize = 0;

    while let Some((scan_source_idx, mut file_reader, row_deletions)) =
        readers_init_iter.next().await.transpose()?
    {
        let n_rows = file_reader.n_rows_in_file().await?;

        let n_rows_deleted = if let Some(row_deletions) = row_deletions {
            let mask = row_deletions.into_external_filter_mask().await?;
            let n_rows_deleted = mask.num_deleted_rows();

            file_row_deletions.insert(scan_source_idx, mask.clone());

            n_rows_deleted
        } else {
            0
        };

        let n_rows_this_file = RowCounter::new(n_rows, n_rows_deleted);

        // push_front: we are walking in reverse
        initialized_readers.push_front((file_reader, n_rows_this_file));

        n_rows_seen = n_rows_seen.add(n_rows_this_file);
        n_files_from_end += 1;

        // Trim readers from end that are already past slice_len.
        while initialized_readers.len() > 1 {
            // `current_rows_held` we exclude the latest file, as the slice offset could begin
            // from any position within that file, meaning that the file could have extra rows
            // that do not contribute to the slice_len.
            let current_rows_held = n_rows_seen.sub(n_rows_this_file.add(n_rows_trimmed));
            let extra_rows_held = current_rows_held.num_rows()?.saturating_sub(slice_len);

            if extra_rows_held < initialized_readers.back().unwrap().1.num_rows()? {
                break;
            }

            let (_reader, row_counter) = initialized_readers.pop_back().unwrap();

            n_rows_trimmed = n_rows_trimmed.add(row_counter)
        }

        if n_rows_seen.num_rows()? >= n_rows_needed {
            break;
        }
    }

    let scan_source_idx = config.sources.len() - n_files_from_end;
    let initialized_readers = Some((scan_source_idx, initialized_readers));
    let resolved_slice = pre_slice.restrict_to_bounds(n_rows_seen.num_rows()?);

    if verbose {
        eprintln!(
            "resolve_negative_slice(): \
            resolved to {resolved_slice:?}, \
            n_rows_seen: {n_rows_seen:?}"
        );
    }

    if slice_len == 0 {
        return Ok(ResolvedSliceInfo {
            scan_source_idx: config.sources.len(),
            row_index: config.row_index.clone(),
            pre_slice: Some(Slice::Positive { offset: 0, len: 0 }),
            initialized_readers: None,
            row_deletions: Default::default(),
        });
    }

    let mut row_index = config.row_index.clone();

    if let Some(row_index) = row_index.as_mut() {
        if verbose {
            eprintln!("resolve_negative_slice(): continuing scan to resolve row index");
        }

        let mut n_rows_skipped_from_start = RowCounter::default();

        // Fully traverse to the beginning to update the row index offset.
        while let Some((_scan_source_idx, mut reader, row_deletions)) =
            readers_init_iter.next().await.transpose()?
        {
            let n_rows = reader.n_rows_in_file().await?;

            let row_deletions = if let Some(row_deletions) = row_deletions {
                Some(row_deletions.into_external_filter_mask().await?)
            } else {
                None
            };

            let n_rows_deleted = row_deletions.map_or(0, |x| x.num_deleted_rows());

            n_rows_skipped_from_start =
                n_rows_skipped_from_start.add(RowCounter::new(n_rows, n_rows_deleted));
        }

        row_index.offset = row_index
            .offset
            .saturating_add(n_rows_skipped_from_start.num_rows_idxsize_saturating()?);
    }

    Ok(ResolvedSliceInfo {
        scan_source_idx,
        row_index,
        pre_slice: Some(resolved_slice),
        initialized_readers,
        row_deletions: file_row_deletions,
    })
}
