use std::sync::Arc;

use arrow::datatypes::ArrowSchemaRef;
use async_trait::async_trait;
use polars_core::prelude::ArrowSchema;
use polars_core::schema::{Schema, SchemaExt, SchemaRef};
use polars_error::{PolarsResult, polars_err};
use polars_io::cloud::CloudOptions;
use polars_io::predicates::ScanIOPredicate;
use polars_io::prelude::{FileMetadata, ParquetOptions};
use polars_io::utils::byte_source::{BufferByteSource, DynByteSource, DynByteSourceBuilder};
use polars_io::{RowIndex, pl_async};
use polars_parquet::read::schema::infer_schema_with_options;
use polars_plan::dsl::ScanSource;
use polars_utils::IdxSize;
use polars_utils::mem::prefetch::get_memory_prefetch_func;
use polars_utils::slice_enum::Slice;

use super::multi_scan::reader_interface::output::{FileReaderOutputRecv, FileReaderOutputSend};
use super::multi_scan::reader_interface::{
    BeginReadArgs, FileReader, FileReaderCallbacks, calc_row_position_after_slice,
};
use crate::async_executor::{self};
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::metrics::OptIOMetrics;
use crate::morsel::SourceToken;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::io_sources::parquet::projection::{
    ArrowFieldProjection, resolve_arrow_field_projections,
};
use crate::nodes::{TaskPriority, io_sources};
use crate::utils::tokio_handle_ext;

pub mod builder;
pub mod init;
mod metadata_utils;
mod projection;
mod row_group_data_fetch;
mod row_group_decode;
mod statistics;

pub struct ParquetFileReader {
    scan_source: ScanSource,
    cloud_options: Option<Arc<CloudOptions>>,
    config: Arc<ParquetOptions>,
    /// Set by the builder if we have metadata left over from DSL conversion.
    metadata: Option<Arc<FileMetadata>>,
    byte_source_builder: DynByteSourceBuilder,
    row_group_prefetch_sync: RowGroupPrefetchSync,
    io_metrics: OptIOMetrics,
    verbose: bool,

    /// Set during initialize()
    init_data: Option<InitializedState>,
}

struct RowGroupPrefetchSync {
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
    file_schema: Arc<ArrowSchema>,
    file_schema_pl: Option<SchemaRef>,
    byte_source: Arc<DynByteSource>,
}

#[async_trait]
impl FileReader for ParquetFileReader {
    async fn initialize(&mut self) -> PolarsResult<()> {
        let verbose = self.verbose;

        if self.init_data.is_some() {
            return Ok(());
        }

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

        let mut byte_source = Arc::new(byte_source);

        let file_metadata = if let Some(v) = self.metadata.clone() {
            v
        } else {
            let (metadata_bytes, opt_full_bytes) = {
                let byte_source = byte_source.clone();
                let io_metrics = self.io_metrics.clone();

                pl_async::get_runtime()
                    .spawn(async move {
                        metadata_utils::read_parquet_metadata_bytes(
                            &byte_source,
                            verbose,
                            &io_metrics,
                        )
                        .await
                    })
                    .await
                    .unwrap()?
            };

            if let Some(full_bytes) = opt_full_bytes {
                byte_source = Arc::new(DynByteSource::Buffer(BufferByteSource(full_bytes)));
            }

            Arc::new(polars_parquet::parquet::read::deserialize_metadata(
                metadata_bytes.as_ref(),
                metadata_bytes.len() * 2 + 1024,
            )?)
        };

        let file_schema = Arc::new(infer_schema_with_options(&file_metadata, &None)?);

        self.init_data = Some(InitializedState {
            file_metadata,
            file_schema,
            file_schema_pl: None,
            byte_source,
        });

        Ok(())
    }

    fn prepare_read(&mut self) -> PolarsResult<()> {
        let wait_group_this_reader = WaitGroup::default();
        let prefetch_all_spawned_token = wait_group_this_reader.token();

        let prev_wait_group: Option<WaitGroup> = self
            .row_group_prefetch_sync
            .shared_prefetch_wait_group_slot
            .try_lock()
            .unwrap()
            .replace(wait_group_this_reader);

        self.row_group_prefetch_sync.prev_all_spawned = prev_wait_group;
        self.row_group_prefetch_sync.current_all_spawned = Some(prefetch_all_spawned_token);

        Ok(())
    }

    fn begin_read(
        &mut self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)> {
        let verbose = self.verbose;

        let InitializedState {
            file_metadata,
            file_schema: file_arrow_schema,
            file_schema_pl: _,
            byte_source,
        } = self.init_data.clone().unwrap();

        let n_rows_in_file = self._n_rows_in_file()?;

        let single_morsel_height: Option<usize> = if let BeginReadArgs {
            projection,
            row_index: None,
            pre_slice,
            predicate: None,
            cast_columns_policy: _,
            num_pipelines: _,
            disable_morsel_split: true,
            callbacks:
                FileReaderCallbacks {
                    file_schema_tx: _,
                    n_rows_in_file_tx: _,
                    row_position_on_end_tx: _,
                },
        } = &args
            && projection.is_empty()
        {
            let mut h: usize = n_rows_in_file as _;

            if let Some(pre_slice) = pre_slice.clone() {
                h = usize::min(h, pre_slice.restrict_to_bounds(h).len());
            }

            Some(h)
        } else {
            None
        };

        let BeginReadArgs {
            projection,
            row_index,
            pre_slice: pre_slice_arg,
            predicate,
            cast_columns_policy,
            num_pipelines,
            disable_morsel_split,
            callbacks:
                FileReaderCallbacks {
                    file_schema_tx,
                    n_rows_in_file_tx,
                    row_position_on_end_tx,
                },
        } = args;

        let file_schema = self._file_schema().clone();

        let normalized_pre_slice = pre_slice_arg
            .clone()
            .map(|x| x.restrict_to_bounds(usize::try_from(n_rows_in_file).unwrap()));

        // Send all callbacks to unblock the next reader. We can do this immediately as we know
        // the total row count upfront.

        if let Some(n_rows_in_file_tx) = n_rows_in_file_tx {
            _ = n_rows_in_file_tx.send(n_rows_in_file);
        }

        // We are allowed to send this value immediately, even though we haven't "ended" yet
        // (see its definition under FileReaderCallbacks).
        if let Some(row_position_on_end_tx) = row_position_on_end_tx {
            _ = row_position_on_end_tx
                .send(self._row_position_after_slice(normalized_pre_slice.clone())?);
        }

        if let Some(file_schema_tx) = file_schema_tx {
            _ = file_schema_tx.send(file_schema.clone());
        }

        if normalized_pre_slice.as_ref().is_some_and(|x| x.len() == 0) {
            let (_, rx) = FileReaderOutputSend::new_serial();

            if verbose {
                eprintln!(
                    "[ParquetFileReader]: early return: \
                    n_rows_in_file: {n_rows_in_file}, \
                    pre_slice: {pre_slice_arg:?}, \
                    resolved_pre_slice: {normalized_pre_slice:?}"
                )
            }

            return Ok((
                rx,
                async_executor::spawn(TaskPriority::Low, std::future::ready(Ok(()))),
            ));
        }

        let mut _projected_arrow_fields: Option<Arc<[ArrowFieldProjection]>> = None;
        let mut projected_arrow_fields = || {
            if _projected_arrow_fields.is_none() {
                _projected_arrow_fields = Some(resolve_arrow_field_projections(
                    &file_arrow_schema,
                    &file_schema,
                    projection.clone(),
                    cast_columns_policy.clone(),
                )?);
            }
            PolarsResult::Ok(_projected_arrow_fields.as_ref().unwrap().clone())
        };

        if verbose {
            eprintln!(
                "[ParquetFileReader]: \
                project: {} / {}, \
                pre_slice: {:?}, \
                resolved_pre_slice: {:?}, \
                row_index: {:?}, \
                predicate: {:?}",
                projected_arrow_fields()?.len(),
                file_schema.len(),
                pre_slice_arg,
                normalized_pre_slice,
                &row_index,
                predicate.as_ref().map(|_| "<predicate>"),
            )
        }

        if let Some(single_morsel_height) = single_morsel_height {
            let (mut tx, rx) = FileReaderOutputSend::new_serial();

            let handle = async_executor::spawn(TaskPriority::Low, async move {
                let _ = tx
                    .send_morsel(Morsel::new(
                        DataFrame::empty_with_height(single_morsel_height),
                        MorselSeq::default(),
                        SourceToken::default(),
                    ))
                    .await;
                Ok(())
            });

            return Ok((rx, handle));
        }

        // Prepare parameters for dispatch
        let projected_arrow_fields = projected_arrow_fields()?.clone();
        let memory_prefetch_func = get_memory_prefetch_func(verbose);
        let row_group_prefetch_size = self
            .row_group_prefetch_sync
            .prefetch_limit
            .min(file_metadata.row_groups.len())
            .max(1);

        // This can be set to 1 to force column-per-thread parallelism, e.g. for bug reproduction.
        let target_values_per_thread =
            std::env::var("POLARS_PARQUET_DECODE_TARGET_VALUES_PER_THREAD")
                .map(|x| x.parse::<usize>().expect("integer").max(1))
                .unwrap_or(16_777_216);

        let is_full_projection = projected_arrow_fields.len() == file_schema.len();

        let (output_recv, handle) = ParquetReadImpl {
            projected_arrow_fields,
            is_full_projection,
            predicate,
            // TODO: Refactor to avoid full clone
            options: Arc::unwrap_or_clone(self.config.clone()),
            byte_source,
            normalized_pre_slice: normalized_pre_slice.map(|x| match x {
                Slice::Positive { offset, len } => (offset, len),
                Slice::Negative { .. } => unreachable!(),
            }),
            metadata: file_metadata,
            config: io_sources::parquet::Config {
                num_pipelines,
                row_group_prefetch_size,
                target_values_per_thread,
            },
            verbose,
            memory_prefetch_func,
            row_index,
            rg_prefetch_semaphore: Arc::clone(&self.row_group_prefetch_sync.prefetch_semaphore),
            rg_prefetch_prev_all_spawned: Option::take(
                &mut self.row_group_prefetch_sync.prev_all_spawned,
            ),
            rg_prefetch_current_all_spawned: Option::take(
                &mut self.row_group_prefetch_sync.current_all_spawned,
            ),
            io_metrics: self.io_metrics.clone(),
            disable_morsel_split,
        }
        .run();

        Ok((
            output_recv,
            async_executor::spawn(TaskPriority::Low, async move { handle.await.unwrap() }),
        ))
    }

    async fn file_schema(&mut self) -> PolarsResult<SchemaRef> {
        Ok(self._file_schema().clone())
    }

    async fn file_arrow_schema(&mut self) -> PolarsResult<Option<ArrowSchemaRef>> {
        Ok(Some(self._file_arrow_schema().clone()))
    }

    async fn n_rows_in_file(&mut self) -> PolarsResult<IdxSize> {
        self._n_rows_in_file()
    }

    async fn fast_n_rows_in_file(&mut self) -> PolarsResult<Option<IdxSize>> {
        self._n_rows_in_file().map(Some)
    }

    async fn row_position_after_slice(
        &mut self,
        pre_slice: Option<Slice>,
    ) -> PolarsResult<IdxSize> {
        self._row_position_after_slice(pre_slice)
    }
}

impl ParquetFileReader {
    fn _file_schema(&mut self) -> &SchemaRef {
        let InitializedState {
            file_schema,
            file_schema_pl,
            ..
        } = self.init_data.as_mut().unwrap();

        if file_schema_pl.is_none() {
            *file_schema_pl = Some(Arc::new(Schema::from_arrow_schema(file_schema.as_ref())))
        }

        file_schema_pl.as_ref().unwrap()
    }

    fn _file_arrow_schema(&mut self) -> &ArrowSchemaRef {
        let InitializedState { file_schema, .. } = self.init_data.as_mut().unwrap();
        file_schema
    }

    fn _n_rows_in_file(&self) -> PolarsResult<IdxSize> {
        let n = self.init_data.as_ref().unwrap().file_metadata.num_rows;
        IdxSize::try_from(n).map_err(|_| polars_err!(bigidx, ctx = "parquet file", size = n))
    }

    fn _row_position_after_slice(&self, pre_slice: Option<Slice>) -> PolarsResult<IdxSize> {
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

struct ParquetReadImpl {
    projected_arrow_fields: Arc<[ArrowFieldProjection]>,
    is_full_projection: bool,
    predicate: Option<ScanIOPredicate>,
    options: ParquetOptions,
    byte_source: Arc<DynByteSource>,
    normalized_pre_slice: Option<(usize, usize)>,
    metadata: Arc<FileMetadata>,
    // Run-time vars
    config: Config,
    verbose: bool,
    memory_prefetch_func: fn(&[u8]) -> (),
    row_index: Option<RowIndex>,

    rg_prefetch_semaphore: Arc<tokio::sync::Semaphore>,
    rg_prefetch_prev_all_spawned: Option<WaitGroup>,
    rg_prefetch_current_all_spawned: Option<WaitToken>,
    io_metrics: OptIOMetrics,
    disable_morsel_split: bool,
}

#[derive(Debug)]
struct Config {
    num_pipelines: usize,
    /// Number of row groups to pre-fetch concurrently, this can be across files
    row_group_prefetch_size: usize,
    /// Minimum number of values for a parallel spawned task to process to amortize
    /// parallelism overhead.
    target_values_per_thread: usize,
}

impl ParquetReadImpl {
    fn run(mut self) -> AsyncTaskData {
        if self.verbose {
            eprintln!("[ParquetFileReader]: {:?}", &self.config);
        }

        self.init_morsel_distributor()
    }
}
