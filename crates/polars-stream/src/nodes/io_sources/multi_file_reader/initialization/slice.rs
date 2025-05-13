use std::collections::VecDeque;

use futures::StreamExt;
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_utils::IdxSize;
use polars_utils::slice_enum::Slice;

use crate::async_executor::{self, AbortOnDropHandle, TaskPriority};
use crate::nodes::io_sources::multi_file_reader::MultiFileReaderConfig;
use crate::nodes::io_sources::multi_file_reader::reader_interface::FileReader;

pub struct ResolvedSliceInfo {
    pub scan_source_idx: usize,
    pub row_index: Option<RowIndex>,
    /// This should always be positive slice.
    pub pre_slice: Option<Slice>,
    /// If we resolved a negative slice we keep the initialized readers here (with a limit). For
    /// Parquet this can save a duplicate metadata fetch/decode.
    ///
    /// This will be in-order - i.e. `pop_front()` corresponds to the next reader.
    #[expect(clippy::type_complexity)]
    pub initialized_readers: Option<(usize, VecDeque<(Box<dyn FileReader>, IdxSize)>)>,
}

pub async fn resolve_to_positive_slice(
    config: &MultiFileReaderConfig,
) -> PolarsResult<ResolvedSliceInfo> {
    match config.pre_slice.clone() {
        None => Ok(ResolvedSliceInfo {
            scan_source_idx: 0,
            row_index: config.row_index.clone(),
            pre_slice: None,
            initialized_readers: None,
        }),

        pre_slice @ Some(Slice::Positive { .. }) => Ok(ResolvedSliceInfo {
            scan_source_idx: 0,
            row_index: config.row_index.clone(),
            pre_slice,
            initialized_readers: None,
        }),

        Some(_) => resolve_negative_slice(config).await,
    }
}

/// Translates a negative slice to positive slice.
async fn resolve_negative_slice(config: &MultiFileReaderConfig) -> PolarsResult<ResolvedSliceInfo> {
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
        });
    }

    let mut initialized_readers = VecDeque::with_capacity(
        config
            .sources
            .len()
            .min(config.num_pipelines().saturating_add(4)),
    );

    let mut readers_init_iter = futures::stream::iter((0..config.sources.len()).rev())
        .map(|scan_source_idx| {
            let sources = config.sources.clone();
            let cloud_options = config.cloud_options.clone();
            let file_reader_builder = config.file_reader_builder.clone();

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
                    eprintln!(
                        "resolve_negative_slice(): init scan source {}",
                        scan_source_idx
                    );
                }

                reader.initialize().await?;
                PolarsResult::Ok(reader)
            }))
        })
        .buffered(config.n_readers_pre_init());

    let n_rows_needed = IdxSize::try_from(offset_from_end).unwrap();
    let slice_len_idxsize = IdxSize::try_from(slice_len).unwrap_or(IdxSize::MAX);

    let mut n_rows_seen: IdxSize = 0;
    let mut n_rows_trimmed: IdxSize = 0;
    let mut n_files_from_end: usize = 0;

    while let Some(mut file_reader) = readers_init_iter.next().await.transpose()? {
        let n_rows = file_reader.n_rows_in_file().await?;

        // push_front: we are walking in reverse
        initialized_readers.push_front((file_reader, n_rows));

        n_rows_seen = n_rows_seen.saturating_add(n_rows);
        n_files_from_end += 1;

        let n_rows_this_file = n_rows;

        // Trim readers from end that are already past slice_len.
        while initialized_readers.len() > 1 {
            // `current_rows_held` we exclude the latest file, as the slice offset could begin
            // from any position within that file, meaning that the file could have extra rows
            // that do not contribute to the slice_len.
            let current_rows_held = n_rows_seen - n_rows_trimmed.saturating_add(n_rows_this_file);
            let extra_rows_held = current_rows_held.saturating_sub(slice_len_idxsize);

            if extra_rows_held < initialized_readers.back().unwrap().1 {
                break;
            }

            n_rows_trimmed =
                n_rows_trimmed.saturating_add(initialized_readers.pop_back().unwrap().1)
        }

        if n_rows_seen >= n_rows_needed {
            break;
        }
    }

    let scan_source_idx = config.sources.len() - n_files_from_end;
    let initialized_readers = Some((scan_source_idx, initialized_readers));
    let resolved_slice = pre_slice.restrict_to_bounds(usize::try_from(n_rows_seen).unwrap());

    if verbose {
        eprintln!(
            "resolve_negative_slice(): resolved to {:?}",
            &resolved_slice,
        );
    }

    if slice_len == 0 {
        return Ok(ResolvedSliceInfo {
            scan_source_idx: config.sources.len(),
            row_index: config.row_index.clone(),
            pre_slice: Some(Slice::Positive { offset: 0, len: 0 }),
            initialized_readers: None,
        });
    }

    let mut row_index = config.row_index.clone();

    if let Some(row_index) = row_index.as_mut() {
        if verbose {
            eprintln!("resolve_negative_slice(): continuing scan to resolve row index");
        }

        let mut n_rows_skipped_from_start: IdxSize = 0;

        // Fully traverse to the beginning to update the row index offset.
        while let Some(mut reader) = readers_init_iter.next().await.transpose()? {
            let n_rows = reader.n_rows_in_file().await?;
            n_rows_skipped_from_start = n_rows_skipped_from_start.saturating_add(n_rows);
        }

        row_index.offset = row_index.offset.saturating_add(n_rows_skipped_from_start);
    }

    Ok(ResolvedSliceInfo {
        scan_source_idx,
        row_index,
        pre_slice: Some(resolved_slice),
        initialized_readers,
    })
}
