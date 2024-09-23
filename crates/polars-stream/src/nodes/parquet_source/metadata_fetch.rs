use std::sync::Arc;

use futures::StreamExt;
use polars_error::{polars_bail, PolarsResult};
use polars_io::prelude::FileMetadata;
use polars_io::utils::byte_source::{DynByteSource, MemSliceByteSource};
use polars_io::utils::slice::SplitSlicePosition;
use polars_utils::mmap::MemSlice;
use polars_utils::pl_str::PlSmallStr;

use super::metadata_utils::{ensure_schema_has_projected_fields, read_parquet_metadata_bytes};
use super::ParquetSourceNode;
use crate::async_executor;
use crate::async_primitives::connector::connector;
use crate::nodes::TaskPriority;
use crate::utils::task_handles_ext;

impl ParquetSourceNode {
    /// Constructs the task that fetches file metadata.
    /// Note: This must be called AFTER `self.projected_arrow_schema` has been initialized.
    #[allow(clippy::type_complexity)]
    pub(super) fn init_metadata_fetcher(
        &mut self,
    ) -> (
        tokio::sync::oneshot::Receiver<Option<(usize, usize)>>,
        crate::async_primitives::connector::Receiver<(
            usize,
            usize,
            Arc<DynByteSource>,
            FileMetadata,
        )>,
        task_handles_ext::AbortOnDropHandle<PolarsResult<()>>,
    ) {
        let verbose = self.verbose;
        let io_runtime = polars_io::pl_async::get_runtime();

        let projected_arrow_schema = self.projected_arrow_schema.clone().unwrap();

        let (normalized_slice_oneshot_tx, normalized_slice_oneshot_rx) =
            tokio::sync::oneshot::channel();
        let (mut metadata_tx, metadata_rx) = connector();

        let byte_source_builder = self.byte_source_builder.clone();

        if self.verbose {
            eprintln!(
                "[ParquetSource]: Byte source builder: {:?}",
                &byte_source_builder
            );
        }

        let fetch_metadata_bytes_for_path_index = {
            let scan_sources = &self.scan_sources;
            let cloud_options = Arc::new(self.cloud_options.clone());

            let scan_sources = scan_sources.clone();
            let cloud_options = cloud_options.clone();
            let byte_source_builder = byte_source_builder.clone();
            let have_first_metadata = self.first_metadata.is_some();

            move |path_idx: usize| {
                let scan_sources = scan_sources.clone();
                let cloud_options = cloud_options.clone();
                let byte_source_builder = byte_source_builder.clone();

                let handle = io_runtime.spawn(async move {
                    let mut byte_source = Arc::new(
                        scan_sources
                            .get(path_idx)
                            .unwrap()
                            .to_dyn_byte_source(
                                &byte_source_builder,
                                cloud_options.as_ref().as_ref(),
                            )
                            .await?,
                    );

                    if path_idx == 0 && have_first_metadata {
                        let metadata_bytes = MemSlice::EMPTY;
                        return Ok((0, byte_source, metadata_bytes));
                    }

                    let (metadata_bytes, maybe_full_bytes) =
                        read_parquet_metadata_bytes(byte_source.as_ref(), verbose).await?;

                    if let Some(v) = maybe_full_bytes {
                        if !matches!(byte_source.as_ref(), DynByteSource::MemSlice(_)) {
                            if verbose {
                                eprintln!(
                                    "[ParquetSource]: Parquet file was fully fetched during \
                                         metadata read ({} bytes).",
                                    v.len(),
                                );
                            }

                            byte_source = Arc::new(DynByteSource::from(MemSliceByteSource(v)))
                        }
                    }

                    PolarsResult::Ok((path_idx, byte_source, metadata_bytes))
                });

                let handle = task_handles_ext::AbortOnDropHandle(handle);

                std::future::ready(handle)
            }
        };

        let first_metadata = self.first_metadata.clone();
        let reader_schema_len = self
            .file_info
            .reader_schema
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap_left()
            .len();
        let has_projection = self.file_options.with_columns.is_some();

        let process_metadata_bytes = {
            move |handle: task_handles_ext::AbortOnDropHandle<
                PolarsResult<(usize, Arc<DynByteSource>, MemSlice)>,
            >| {
                let projected_arrow_schema = projected_arrow_schema.clone();
                let first_metadata = first_metadata.clone();
                // Run on CPU runtime - metadata deserialization is expensive, especially
                // for very wide tables.
                let handle = async_executor::spawn(TaskPriority::Low, async move {
                    let (path_index, byte_source, metadata_bytes) = handle.await.unwrap()?;

                    let metadata = match first_metadata {
                        Some(md) if path_index == 0 => Arc::unwrap_or_clone(md),
                        _ => polars_parquet::parquet::read::deserialize_metadata(
                            metadata_bytes.as_ref(),
                            metadata_bytes.len() * 2 + 1024,
                        )?,
                    };

                    let schema = polars_parquet::arrow::read::infer_schema(&metadata)?;

                    if !has_projection && schema.len() > reader_schema_len {
                        polars_bail!(
                           SchemaMismatch:
                           "parquet file contained extra columns and no selection was given"
                        )
                    }

                    ensure_schema_has_projected_fields(&schema, projected_arrow_schema.as_ref())?;

                    PolarsResult::Ok((path_index, byte_source, metadata))
                });

                async_executor::AbortOnDropHandle::new(handle)
            }
        };

        let metadata_prefetch_size = self.config.metadata_prefetch_size;
        let metadata_decode_ahead_size = self.config.metadata_decode_ahead_size;

        let (start_tx, start_rx) = tokio::sync::oneshot::channel();
        self.morsel_stream_starter = Some(start_tx);

        let metadata_task_handle = if self
            .file_options
            .slice
            .map(|(offset, _)| offset >= 0)
            .unwrap_or(true)
        {
            normalized_slice_oneshot_tx
                .send(
                    self.file_options
                        .slice
                        .map(|(offset, len)| (offset as usize, len)),
                )
                .unwrap();

            // Safety: `offset + len` does not overflow.
            let slice_range = self
                .file_options
                .slice
                .map(|(offset, len)| offset as usize..offset as usize + len);

            let mut metadata_stream = futures::stream::iter(0..self.scan_sources.len())
                .map(fetch_metadata_bytes_for_path_index)
                .buffered(metadata_prefetch_size)
                .map(process_metadata_bytes)
                .buffered(metadata_decode_ahead_size);

            let scan_sources = self.scan_sources.clone();

            // We need to be able to both stop early as well as skip values, which is easier to do
            // using a custom task instead of futures::stream
            io_runtime.spawn(async move {
                let current_row_offset_ref = &mut 0usize;
                let current_path_index_ref = &mut 0usize;

                if start_rx.await.is_err() {
                    return Ok(());
                }

                if verbose {
                    eprintln!("[ParquetSource]: Starting data fetch")
                }

                loop {
                    let current_path_index = *current_path_index_ref;
                    *current_path_index_ref += 1;

                    let Some(v) = metadata_stream.next().await else {
                        break;
                    };

                    let (path_index, byte_source, metadata) = v.map_err(|err| {
                        err.wrap_msg(|msg| {
                            format!(
                                "error at path (index: {}, path: {:?}): {}",
                                current_path_index,
                                scan_sources
                                    .get(current_path_index)
                                    .map(|x| PlSmallStr::from_str(x.to_include_path_name())),
                                msg
                            )
                        })
                    })?;

                    assert_eq!(path_index, current_path_index);

                    let current_row_offset = *current_row_offset_ref;
                    *current_row_offset_ref = current_row_offset.saturating_add(metadata.num_rows);

                    if let Some(slice_range) = slice_range.clone() {
                        match SplitSlicePosition::split_slice_at_file(
                            current_row_offset,
                            metadata.num_rows,
                            slice_range,
                        ) {
                            SplitSlicePosition::Before => {
                                if verbose {
                                    eprintln!(
                                        "[ParquetSource]: Slice pushdown: \
                                            Skipped file at index {} ({} rows)",
                                        current_path_index, metadata.num_rows
                                    );
                                }
                                continue;
                            },
                            SplitSlicePosition::After => unreachable!(),
                            SplitSlicePosition::Overlapping(..) => {},
                        };
                    };

                    if metadata_tx
                        .send((path_index, current_row_offset, byte_source, metadata))
                        .await
                        .is_err()
                    {
                        break;
                    }

                    if let Some(slice_range) = slice_range.as_ref() {
                        if *current_row_offset_ref >= slice_range.end {
                            if verbose {
                                eprintln!(
                                    "[ParquetSource]: Slice pushdown: \
                                        Stopped reading at file at index {} \
                                        (remaining {} files will not be read)",
                                    current_path_index,
                                    scan_sources.len() - current_path_index - 1,
                                );
                            }
                            break;
                        }
                    };
                }

                Ok(())
            })
        } else {
            // Walk the files in reverse to translate the slice into a positive offset.
            let slice = self.file_options.slice.unwrap();
            let slice_start_as_n_from_end = -slice.0 as usize;

            let mut metadata_stream = futures::stream::iter((0..self.scan_sources.len()).rev())
                .map(fetch_metadata_bytes_for_path_index)
                .buffered(metadata_prefetch_size)
                .map(process_metadata_bytes)
                .buffered(metadata_decode_ahead_size);

            // Note:
            // * We want to wait until the first morsel is requested before starting this
            let init_negative_slice_and_metadata = async move {
                let mut processed_metadata_rev = vec![];
                let mut cum_rows = 0;

                while let Some(v) = metadata_stream.next().await {
                    let v = v?;
                    let (_, _, metadata) = &v;
                    cum_rows += metadata.num_rows;
                    processed_metadata_rev.push(v);

                    if cum_rows >= slice_start_as_n_from_end {
                        break;
                    }
                }

                let (start, len) = if slice_start_as_n_from_end > cum_rows {
                    // We need to trim the slice, e.g. SLICE[offset: -100, len: 75] on a file of 50
                    // rows should only give the first 25 rows.
                    let first_file_position = slice_start_as_n_from_end - cum_rows;
                    (0, slice.1.saturating_sub(first_file_position))
                } else {
                    (cum_rows - slice_start_as_n_from_end, slice.1)
                };

                if len == 0 {
                    processed_metadata_rev.clear();
                }

                normalized_slice_oneshot_tx
                    .send(Some((start, len)))
                    .unwrap();

                let slice_range = start..(start + len);

                PolarsResult::Ok((slice_range, processed_metadata_rev, cum_rows))
            };

            let path_count = self.scan_sources.len();

            io_runtime.spawn(async move {
                if start_rx.await.is_err() {
                    return Ok(());
                }

                if verbose {
                    eprintln!("[ParquetSource]: Starting data fetch (negative slice)")
                }

                let (slice_range, processed_metadata_rev, cum_rows) =
                    async_executor::AbortOnDropHandle::new(async_executor::spawn(
                        TaskPriority::Low,
                        init_negative_slice_and_metadata,
                    ))
                    .await?;

                if verbose {
                    if let Some((path_index, ..)) = processed_metadata_rev.last() {
                        eprintln!(
                            "[ParquetSource]: Slice pushdown: Negatively-offsetted slice {:?} \
                            begins at file index {}, translated to {:?}",
                            slice, path_index, slice_range
                        );
                    } else {
                        eprintln!(
                            "[ParquetSource]: Slice pushdown: Negatively-offsetted slice {:?} \
                            skipped all files ({} files containing {} rows)",
                            slice, path_count, cum_rows
                        )
                    }
                }

                let metadata_iter = processed_metadata_rev.into_iter().rev();
                let current_row_offset_ref = &mut 0usize;

                for (current_path_index, byte_source, metadata) in metadata_iter {
                    let current_row_offset = *current_row_offset_ref;
                    *current_row_offset_ref = current_row_offset.saturating_add(metadata.num_rows);

                    assert!(matches!(
                        SplitSlicePosition::split_slice_at_file(
                            current_row_offset,
                            metadata.num_rows,
                            slice_range.clone(),
                        ),
                        SplitSlicePosition::Overlapping(..)
                    ));

                    if metadata_tx
                        .send((
                            current_path_index,
                            current_row_offset,
                            byte_source,
                            metadata,
                        ))
                        .await
                        .is_err()
                    {
                        break;
                    }

                    if *current_row_offset_ref >= slice_range.end {
                        if verbose {
                            eprintln!(
                                "[ParquetSource]: Slice pushdown: \
                                Stopped reading at file at index {} \
                                (remaining {} files will not be read)",
                                current_path_index,
                                path_count - current_path_index - 1,
                            );
                        }
                        break;
                    }
                }

                Ok(())
            })
        };

        let metadata_task_handle = task_handles_ext::AbortOnDropHandle(metadata_task_handle);

        (
            normalized_slice_oneshot_rx,
            metadata_rx,
            metadata_task_handle,
        )
    }
}
