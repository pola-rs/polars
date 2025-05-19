use std::sync::Arc;

use arrow::datatypes::ArrowSchemaRef;
use async_trait::async_trait;
use polars_core::prelude::ArrowSchema;
use polars_core::schema::{Schema, SchemaExt, SchemaRef};
use polars_error::{PolarsResult, polars_err};
use polars_io::cloud::CloudOptions;
use polars_io::predicates::ScanIOPredicate;
use polars_io::prelude::{FileMetadata, ParquetOptions};
use polars_io::utils::byte_source::{DynByteSource, DynByteSourceBuilder, MemSliceByteSource};
use polars_io::{RowIndex, pl_async};
use polars_parquet::read::schema::infer_schema_with_options;
use polars_plan::dsl::{CastColumnsPolicy, ScanSource};
use polars_utils::IdxSize;
use polars_utils::mem::prefetch::get_memory_prefetch_func;
use polars_utils::slice_enum::Slice;

use super::multi_file_reader::extra_ops::cast_columns::CastColumns;
use super::multi_file_reader::reader_interface::output::{
    FileReaderOutputRecv, FileReaderOutputSend,
};
use super::multi_file_reader::reader_interface::{
    BeginReadArgs, FileReader, FileReaderCallbacks, calc_row_position_after_slice,
};
use crate::async_executor::{self};
use crate::nodes::compute_node_prelude::*;
use crate::nodes::{TaskPriority, io_sources};
use crate::utils::task_handles_ext;

pub mod builder;
mod init;
mod metadata_utils;
mod row_group_data_fetch;
mod row_group_decode;

pub struct ParquetFileReader {
    scan_source: ScanSource,
    cloud_options: Option<Arc<CloudOptions>>,
    config: Arc<ParquetOptions>,
    /// Set by the builder if we have metadata left over from DSL conversion.
    metadata: Option<Arc<FileMetadata>>,
    byte_source_builder: DynByteSourceBuilder,
    verbose: bool,

    /// Set during initialize()
    init_data: Option<InitializedState>,
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

                pl_async::get_runtime()
                    .spawn(async move {
                        metadata_utils::read_parquet_metadata_bytes(&byte_source, verbose).await
                    })
                    .await
                    .unwrap()?
            };

            if let Some(full_bytes) = opt_full_bytes {
                byte_source = Arc::new(DynByteSource::MemSlice(MemSliceByteSource(full_bytes)));
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

    fn begin_read(
        &mut self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)> {
        let verbose = self.verbose;

        let InitializedState {
            file_metadata,
            file_schema,
            file_schema_pl: _,
            byte_source,
        } = self.init_data.clone().unwrap();

        let BeginReadArgs {
            projected_schema,
            row_index,
            pre_slice: pre_slice_arg,
            mut predicate,
            cast_columns_policy,
            num_pipelines,
            callbacks:
                FileReaderCallbacks {
                    file_schema_tx,
                    n_rows_in_file_tx,
                    row_position_on_end_tx,
                },
        } = args;

        let n_rows_in_file = self._n_rows_in_file()?;

        let normalized_pre_slice = pre_slice_arg
            .clone()
            .map(|x| x.restrict_to_bounds(usize::try_from(n_rows_in_file).unwrap()));

        // Send all callbacks to unblock the next reader. We can do this immediately as we know
        // the total row count upfront.

        if let Some(mut n_rows_in_file_tx) = n_rows_in_file_tx {
            _ = n_rows_in_file_tx.try_send(n_rows_in_file);
        }

        // We are allowed to send this value immediately, even though we haven't "ended" yet
        // (see its definition under FileReaderCallbacks).
        if let Some(mut row_position_on_end_tx) = row_position_on_end_tx {
            _ = row_position_on_end_tx
                .try_send(self._row_position_after_slice(normalized_pre_slice.clone())?);
        }

        if let Some(mut file_schema_tx) = file_schema_tx {
            _ = file_schema_tx.try_send(self._file_schema());
        }

        if normalized_pre_slice.as_ref().is_some_and(|x| x.len() == 0) {
            let (_, rx) = FileReaderOutputSend::new_serial();

            if verbose {
                eprintln!(
                    "[ParquetFileReader]: early return: \
                    n_rows_in_file: {} \
                    pre_slice: {:?} \
                    resolved_pre_slice: {:?} \
                    ",
                    n_rows_in_file, pre_slice_arg, normalized_pre_slice
                )
            }

            return Ok((
                rx,
                async_executor::spawn(TaskPriority::Low, std::future::ready(Ok(()))),
            ));
        }

        // Prepare parameters for dispatch

        let memory_prefetch_func = get_memory_prefetch_func(verbose);
        let row_group_prefetch_size = polars_core::config::get_rg_prefetch_size();

        // This can be set to 1 to force column-per-thread parallelism, e.g. for bug reproduction.
        let min_values_per_thread = std::env::var("POLARS_MIN_VALUES_PER_THREAD")
            .map(|x| x.parse::<usize>().expect("integer").max(1))
            .unwrap_or(16_777_216);

        let projected_arrow_schema: ArrowSchemaRef = Arc::new(
            projected_schema
                .iter_names()
                .filter(|name| file_schema.contains(name))
                .map(|name| (name.clone(), file_schema.get(name).unwrap().clone()))
                .collect(),
        );

        if verbose {
            eprintln!(
                "[ParquetFileReader]: \
                project: {} / {}, \
                pre_slice: {:?}, \
                resolved_pre_slice: {:?}, \
                row_index: {:?}, \
                predicate: {:?} \
                ",
                projected_arrow_schema.len(),
                file_schema.len(),
                pre_slice_arg,
                normalized_pre_slice,
                &row_index,
                predicate.as_ref().map(|_| "<predicate>"),
            )
        }

        // If are handling predicates we apply missing / cast columns policy here as those need to
        // happen before filtering. Otherwise we leave it to post.
        if let Some(predicate) = predicate.as_mut() {
            if cast_columns_policy != CastColumnsPolicy::ERROR_ON_MISMATCH {
                unimplemented!("column casting w/ predicate in parquet")
            }

            // Note: This currently could return a Some(_) for the case where
            // there are struct fields ordered differently, but that should not
            // affect predicates.
            CastColumns::try_init_from_policy_from_iter(
                &cast_columns_policy,
                &projected_schema,
                &mut self
                    ._file_schema()
                    .iter()
                    .filter(|(name, _)| predicate.live_columns.contains(*name))
                    .map(|(name, dtype)| (name.as_ref(), dtype)),
            )?;
        }

        let (output_recv, handle) = ParquetReadImpl {
            predicate,
            // TODO: Refactor to avoid full clone
            options: Arc::unwrap_or_clone(self.config.clone()),
            byte_source: byte_source.clone(),
            normalized_pre_slice: normalized_pre_slice.map(|x| match x {
                Slice::Positive { offset, len } => (offset, len),
                Slice::Negative { .. } => unreachable!(),
            }),
            metadata: file_metadata.clone(),
            config: io_sources::parquet::Config {
                num_pipelines,
                row_group_prefetch_size,
                min_values_per_thread,
            },
            verbose,
            schema: file_schema.clone(),
            projected_arrow_schema,
            memory_prefetch_func,
            row_index,
        }
        .run();

        Ok((
            output_recv,
            async_executor::spawn(TaskPriority::Low, async move { handle.await.unwrap() }),
        ))
    }

    async fn file_schema(&mut self) -> PolarsResult<SchemaRef> {
        Ok(self._file_schema())
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

impl ParquetFileReader {
    fn _file_schema(&mut self) -> SchemaRef {
        let InitializedState {
            file_schema,
            file_schema_pl,
            ..
        } = self.init_data.as_mut().unwrap();

        if file_schema_pl.is_none() {
            *file_schema_pl = Some(Arc::new(Schema::from_arrow_schema(file_schema.as_ref())))
        }

        file_schema_pl.clone().unwrap()
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
    task_handles_ext::AbortOnDropHandle<PolarsResult<()>>,
);

struct ParquetReadImpl {
    predicate: Option<ScanIOPredicate>,
    options: ParquetOptions,
    byte_source: Arc<DynByteSource>,
    normalized_pre_slice: Option<(usize, usize)>,
    metadata: Arc<FileMetadata>,
    // Run-time vars
    config: Config,
    verbose: bool,
    schema: Arc<ArrowSchema>,
    projected_arrow_schema: Arc<ArrowSchema>,
    memory_prefetch_func: fn(&[u8]) -> (),
    row_index: Option<RowIndex>,
}

#[derive(Debug)]
struct Config {
    num_pipelines: usize,
    /// Number of row groups to pre-fetch concurrently, this can be across files
    row_group_prefetch_size: usize,
    /// Minimum number of values for a parallel spawned task to process to amortize
    /// parallelism overhead.
    min_values_per_thread: usize,
}

impl ParquetReadImpl {
    fn run(mut self) -> AsyncTaskData {
        if self.verbose {
            eprintln!("[ParquetFileReader]: {:?}", &self.config);
        }

        self.init_morsel_distributor()
    }
}
