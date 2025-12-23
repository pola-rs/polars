use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;

use arrow::io::ipc::read::{Dictionaries, read_dictionary_block};
use async_trait::async_trait;
use polars_core::schema::{Schema, SchemaExt};
use polars_core::utils::arrow::io::ipc::read::{
    BlockReader, FileMetadata, ProjectionInfo, get_row_count_from_blocks, prepare_projection,
    read_file_metadata,
};
use polars_error::{ErrString, PolarsError, PolarsResult, feature_gated, polars_err};
use polars_io::cloud::CloudOptions;
#[cfg(feature = "cloud")]
use polars_io::pl_async::get_runtime;
use polars_io::utils::byte_source::{
    ByteSource, DynByteSource, DynByteSourceBuilder, MemSliceByteSource,
};
use polars_io::{RowIndex, pl_async};
use polars_plan::dsl::{ScanSource, ScanSourceRef};
use polars_utils::IdxSize;
use polars_utils::mem::prefetch::get_memory_prefetch_func;
use polars_utils::plpath::PlPathRef;
use polars_utils::slice_enum::Slice;

use super::multi_scan::reader_interface::output::FileReaderOutputRecv;
use super::multi_scan::reader_interface::{BeginReadArgs, calc_row_position_after_slice};
use crate::async_executor::{self, JoinHandle, TaskPriority, spawn};
use crate::async_primitives::oneshot_channel::Sender;
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::nodes::io_sources::multi_scan::reader_interface::output::FileReaderOutputSend;
use crate::nodes::io_sources::multi_scan::reader_interface::{
    FileReader, FileReaderCallbacks, Projection,
};
use crate::utils::tokio_handle_ext;

pub mod builder;
mod init;
mod record_batch_data_fetch;
mod record_batch_decode;

const ROW_COUNT_OVERFLOW_ERR: PolarsError = PolarsError::ComputeError(ErrString::new_static(
    "\
IPC file produces more than 2^32 rows; \
consider compiling with polars-bigidx feature (pip install polars[rt64])",
));

struct IpcFileReader {
    scan_source: ScanSource,
    cloud_options: Option<Arc<CloudOptions>>,
    metadata: Option<Arc<FileMetadata>>,
    byte_source_builder: DynByteSourceBuilder,
    record_batch_prefetch_sync: RecordBatchPrefetchSync,
    verbose: bool,
    init_data: Option<InitializedState>,
}

struct RecordBatchPrefetchSync {
    prefetch_limit: usize,
    prefetch_semaphore: Arc<tokio::sync::Semaphore>,
    shared_prefetch_wait_group_slot: Arc<std::sync::Mutex<Option<WaitGroup>>>,

    /// Waits for the previous reader to finish spawning prefetches.
    prev_all_spawned: Option<WaitGroup>,
    /// Dropped once the current reader has finished spawning prefetches.
    current_all_spawned: Option<WaitToken>,
}

#[derive(Clone)]
struct InitializedState {
    file_metadata: Arc<FileMetadata>,
    byte_source: Arc<DynByteSource>,
    dictionaries: Arc<Option<Dictionaries>>,
    // Lazily initialized - getting this involves iterating record batches.
    n_rows_in_file: Option<IdxSize>,
}

#[async_trait]
impl FileReader for IpcFileReader {
    async fn initialize(&mut self) -> PolarsResult<()> {
        dbg!("start initialize for impl FileReader for IpcFileReader"); //kdn
        if self.init_data.is_some() {
            return Ok(());
        }

        let verbose = self.verbose;
        let scan_source = self.scan_source.clone();
        let byte_source_builder = self.byte_source_builder.clone();
        let cloud_options = self.cloud_options.clone();

        let byte_source = pl_async::get_runtime()
            .spawn(async move {
                scan_source
                    .as_scan_source_ref()
                    .to_dyn_byte_source(&byte_source_builder, cloud_options.as_deref())
                    .await
            })
            .await
            .unwrap()?;

        let byte_source = Arc::new(byte_source);

        let file_metadata = if let Some(v) = self.metadata.clone() {
            v
        } else {
            self.fetch_file_metadata().await?
        };

        let dictionaries = {
            let byte_source_async = byte_source.clone();
            let metadata_async = file_metadata.clone();
            let dictionaries = pl_async::get_runtime()
                .spawn(async move {
                    read_dictionaries(&byte_source_async, metadata_async, verbose).await
                })
                .await
                .unwrap()?;
            Arc::new(Some(dictionaries))
        };

        self.init_data = Some(InitializedState {
            file_metadata,
            byte_source,
            dictionaries,
            n_rows_in_file: None,
        });
        dbg!("done initialize for impl FileReader for IpcFileReader"); //kdn
        Ok(())
    }

    fn prepare_read(&mut self) -> PolarsResult<()> {
        let wait_group_this_reader = WaitGroup::default();
        let prefetch_all_spawned_token = wait_group_this_reader.token();

        let prev_wait_group: Option<WaitGroup> = self
            .record_batch_prefetch_sync
            .shared_prefetch_wait_group_slot
            .try_lock()
            .unwrap()
            .replace(wait_group_this_reader);

        self.record_batch_prefetch_sync.prev_all_spawned = prev_wait_group;
        self.record_batch_prefetch_sync.current_all_spawned = Some(prefetch_all_spawned_token);

        Ok(())
    }

    fn begin_read(
        &mut self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)> {
        dbg!("start begin_read"); //kdn
        let verbose = self.verbose;

        // Initialize.
        let InitializedState {
            file_metadata,
            byte_source,
            dictionaries,
            n_rows_in_file: _,
        } = self.init_data.clone().unwrap();

        let BeginReadArgs {
            projection: Projection::Plain(projected_schema),
            row_index,
            pre_slice: pre_slice_arg,
            predicate: None,
            cast_columns_policy: _,
            num_pipelines,
            callbacks:
                FileReaderCallbacks {
                    file_schema_tx,
                    n_rows_in_file_tx,
                    row_position_on_end_tx,
                },
        } = args
        else {
            panic!("unsupported args: {:?}", &args)
        };

        let file_schema_pl = std::cell::LazyCell::new(|| {
            Arc::new(Schema::from_arrow_schema(file_metadata.schema.as_ref()))
        });

        // Handle callbacks that are ready now.
        if let Some(file_schema_tx) = file_schema_tx {
            _ = file_schema_tx.send(file_schema_pl.clone());
        }

        // Normalize slice.
        // @NOTE. When this is expensive (e.g., when downloading from cloud) and not strictly required,
        // we skip this normalization and handle it later. Negative slicing requires normalization.
        let normalized_pre_slice = if let Some(pre_slice) = pre_slice_arg.clone()
            && !(self.scan_source.is_cloud_url() && matches!(pre_slice, Slice::Positive { .. }))
        {
            Some(pre_slice.restrict_to_bounds(usize::try_from(self._n_rows_in_file()?).unwrap()))
        } else {
            pre_slice_arg.clone()
        };

        if normalized_pre_slice.as_ref().is_some_and(|x| x.len() == 0) {
            let (_, rx) = FileReaderOutputSend::new_serial();
            let n_rows = self._n_rows_in_file()?;

            if verbose {
                eprintln!(
                    "[IpcFileReader]: early return: \
                    n_rows_in_file: {}, \
                    pre_slice: {:?}, \
                    resolved_pre_slice: {:?} \
                    ",
                    n_rows, pre_slice_arg, normalized_pre_slice
                )
            }

            // Handle callback.
            if let Some(row_position_on_end_tx) = row_position_on_end_tx {
                _ = row_position_on_end_tx.send(n_rows);
            }
            if let Some(n_rows_in_file_tx) = n_rows_in_file_tx {
                _ = n_rows_in_file_tx.send(n_rows);
            }

            return Ok((rx, spawn(TaskPriority::Low, std::future::ready(Ok(())))));
        }

        // Always create a slice. If no slice was given, just make the biggest slice possible.
        let slice_range: Range<usize> = normalized_pre_slice
            .clone()
            .map_or(0..usize::MAX, Range::<usize>::from);

        // Avoid materializing projection info if we are projecting all the columns of this file.
        let projection_indices: Option<Vec<usize>> = if let Some(first_mismatch_idx) =
            (0..file_metadata.schema.len().min(projected_schema.len())).find(|&i| {
                file_metadata.schema.get_at_index(i).unwrap().0
                    != projected_schema.get_at_index(i).unwrap().0
            }) {
            let mut out = Vec::with_capacity(file_metadata.schema.len());

            out.extend(0..first_mismatch_idx);

            out.extend(
                (first_mismatch_idx..projected_schema.len()).filter_map(|i| {
                    file_metadata
                        .schema
                        .index_of(projected_schema.get_at_index(i).unwrap().0)
                }),
            );

            Some(out)
        } else if file_metadata.schema.len() > projected_schema.len() {
            // Names match up to projected schema len.
            Some((0..projected_schema.len()).collect::<Vec<_>>())
        } else {
            // Name order matches up to `file_metadata.schema.len()`, we are projecting all columns
            // in this file.
            None
        };

        if verbose {
            eprintln!(
                "[IpcFileReader]: \
                project: {} / {}, \
                pre_slice: {:?}, \
                resolved_pre_slice: {:?} \
                ",
                projection_indices
                    .as_ref()
                    .map_or(file_metadata.schema.len(), |x| x.len()),
                file_metadata.schema.len(),
                pre_slice_arg,
                normalized_pre_slice
            )
        }

        let projection_info: Option<ProjectionInfo> =
            projection_indices.map(|indices| prepare_projection(&file_metadata.schema, indices));
        let projection_info = Arc::new(projection_info);

        // Prepare parameters for Prefetch
        let memory_prefetch_func = get_memory_prefetch_func(verbose);

        let record_batch_prefetch_size = self
            .record_batch_prefetch_sync
            .prefetch_limit
            .min(file_metadata.blocks.len())
            .max(1);

        let config = Config {
            num_pipelines,
            record_batch_prefetch_size,
        };

        // Create IpcReadImpl and run.
        let (output_recv, handle) = IpcReadImpl {
            byte_source,
            metadata: file_metadata,
            dictionaries,
            slice_range,
            projection_info,
            row_index,
            n_rows_in_file_tx,
            row_position_on_end_tx,
            config,
            verbose,
            memory_prefetch_func,

            rb_prefetch_semaphore: Arc::clone(&self.record_batch_prefetch_sync.prefetch_semaphore),
            rb_prefetch_prev_all_spawned: Option::take(
                &mut self.record_batch_prefetch_sync.prev_all_spawned,
            ),
            rb_prefetch_current_all_spawned: Option::take(
                &mut self.record_batch_prefetch_sync.current_all_spawned,
            ),
        }
        .run();

        Ok((
            output_recv,
            async_executor::spawn(TaskPriority::Low, async move { handle.await.unwrap() }),
        ))
    }

    async fn n_rows_in_file(&mut self) -> PolarsResult<IdxSize> {
        self._n_rows_in_file()
    }

    async fn row_position_after_slice(
        &mut self,
        pre_slice: Option<Slice>,
    ) -> PolarsResult<IdxSize> {
        self._row_position_after_slice(pre_slice)
    }
}

impl IpcFileReader {
    /// Total number of rows in the IPC File. This can be slow if the underlying
    /// byte_source is an object store with a high number of Record Batches.
    fn _n_rows_in_file(&mut self) -> PolarsResult<IdxSize> {
        dbg!("start _n_rows_in_file"); //kdn
        let InitializedState {
            file_metadata,
            byte_source,
            dictionaries: _,
            n_rows_in_file,
        } = self.init_data.as_mut().unwrap();

        match &**byte_source {
            DynByteSource::MemSlice(MemSliceByteSource(memslice)) => {
                if n_rows_in_file.is_none() {
                    let n_rows: i64 = get_row_count_from_blocks(
                        &mut std::io::Cursor::new(memslice.as_ref()),
                        &file_metadata.blocks,
                    )?;

                    let n_rows = IdxSize::try_from(n_rows)
                        .map_err(|_| polars_err!(bigidx, ctx = "ipc file", size = n_rows))?;

                    *n_rows_in_file = Some(n_rows);
                }

                Ok(n_rows_in_file.unwrap())
            },
            byte_source @ DynByteSource::Cloud(_) => {
                let io_runtime = polars_io::pl_async::get_runtime();
                if n_rows_in_file.is_none() {
                    let mut n_rows = 0;
                    let mut message_scratch = Vec::new();
                    let mut ranges: Vec<_> = file_metadata
                        .blocks
                        .iter()
                        .map(|block| {
                            block.offset as usize
                                ..block.offset as usize
                                    + block.meta_data_length as usize
                                    + block.body_length as usize
                        })
                        .collect();

                    let bytes_map = io_runtime.block_on(byte_source.get_ranges(&mut ranges))?;
                    assert_eq!(bytes_map.len(), ranges.len());

                    for bytes in bytes_map.into_values() {
                        let mut reader = BlockReader::new(Cursor::new(bytes.as_ref()));
                        n_rows += reader.record_batch_num_rows(&mut message_scratch)?;
                    }

                    let n_rows = IdxSize::try_from(n_rows)
                        .map_err(|_| polars_err!(bigidx, ctx = "ipc file", size = n_rows))?;

                    *n_rows_in_file = Some(n_rows);
                }

                Ok(n_rows_in_file.unwrap())
            },
        }
    }

    /// Retrieve file metadata from the source.
    async fn fetch_file_metadata(&self) -> PolarsResult<Arc<FileMetadata>> {
        if self.verbose {
            eprintln!("[IpcFileReader]: fetching file metadata");
        }

        let metadata = match self.scan_source.as_scan_source_ref() {
            ScanSourceRef::Path(path) => match path {
                PlPathRef::Cloud(_) => {
                    feature_gated!("cloud", {
                        get_runtime().block_in_place_on(async {
                            let metadata: PolarsResult<_> = {
                                let reader = polars_io::ipc::IpcReaderAsync::from_uri(
                                    path,
                                    self.cloud_options.as_deref(),
                                )
                                .await?;

                                Ok(reader.metadata().await?)
                            };
                            metadata
                        })?
                    })
                },
                PlPathRef::Local(path) => {
                    // Local file I/O is typically synchronous in Arrow-rs
                    let mut reader = std::io::BufReader::new(polars_utils::open_file(path)?);
                    read_file_metadata(&mut reader)?
                },
            },
            ScanSourceRef::File(file) => {
                let mut reader = std::io::BufReader::new(file);
                read_file_metadata(&mut reader)?
            },
            ScanSourceRef::Buffer(buff) => {
                let mut reader = std::io::Cursor::new(buff);
                read_file_metadata(&mut reader)?
            },
        };

        Ok(Arc::new(metadata))
    }

    fn _row_position_after_slice(&mut self, pre_slice: Option<Slice>) -> PolarsResult<IdxSize> {
        Ok(calc_row_position_after_slice(
            self._n_rows_in_file()?,
            pre_slice,
        ))
    }
}

type AsyncTaskData = (
    FileReaderOutputRecv,
    tokio_handle_ext::AbortOnDropHandle<PolarsResult<()>>,
);

struct IpcReadImpl {
    // Source info
    byte_source: Arc<DynByteSource>,
    metadata: Arc<FileMetadata>,
    // Lazily initialized.
    dictionaries: Arc<Option<Dictionaries>>,
    // Query parameters.
    slice_range: Range<usize>,
    projection_info: Arc<Option<ProjectionInfo>>,
    row_index: Option<RowIndex>,
    // Call-backs.
    n_rows_in_file_tx: Option<Sender<IdxSize>>,
    row_position_on_end_tx: Option<Sender<IdxSize>>,
    // Run-time vars.
    config: Config,
    verbose: bool,
    memory_prefetch_func: fn(&[u8]) -> (),

    rb_prefetch_semaphore: Arc<tokio::sync::Semaphore>,
    rb_prefetch_prev_all_spawned: Option<WaitGroup>,
    rb_prefetch_current_all_spawned: Option<WaitToken>,
}

#[derive(Debug)]
struct Config {
    num_pipelines: usize,
    /// Number of record batches to prefetch concurrently, this can be across files
    record_batch_prefetch_size: usize,
}

async fn read_dictionaries(
    byte_source: &DynByteSource,
    file_metadata: Arc<FileMetadata>,
    verbose: bool,
) -> PolarsResult<Dictionaries> {
    let blocks = if let Some(blocks) = &file_metadata.dictionaries {
        blocks
    } else {
        return Ok(Dictionaries::default());
    };

    if verbose {
        eprintln!("[IpcFileReader]: reading dictionaries ({:?})", blocks.len());
    }

    let mut dictionaries = Dictionaries::default();

    let mut message_scratch = Vec::new();
    let mut dictionary_scratch = Vec::new();

    for block in blocks {
        let range = block.offset as usize
            ..block.offset as usize + block.meta_data_length as usize + block.body_length as usize;
        let bytes = byte_source.get_range(range).await?;

        let mut reader = BlockReader::new(Cursor::new(bytes.as_ref()));

        read_dictionary_block(
            &mut reader.reader,
            file_metadata.as_ref(),
            block,
            true,
            &mut dictionaries,
            &mut message_scratch,
            &mut dictionary_scratch,
        )?;
    }

    Ok(dictionaries)
}
