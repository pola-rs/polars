//! TODO: Rename this folder to `parquet` after moving the original or refactoring.
pub mod builder;

use std::sync::Arc;

use arrow::datatypes::ArrowSchemaRef;
use async_trait::async_trait;
use polars_core::prelude::{AnyValue, ArrowSchema};
use polars_core::scalar::Scalar;
use polars_core::schema::{Schema, SchemaExt};
use polars_error::{PolarsResult, polars_err};
use polars_io::cloud::CloudOptions;
use polars_io::pl_async;
use polars_io::prelude::{FileMetadata, ParquetOptions};
use polars_io::utils::byte_source::{DynByteSource, DynByteSourceBuilder, MemSliceByteSource};
use polars_parquet::read::schema::infer_schema_with_options;
use polars_plan::dsl::ScanSource;
use polars_utils::IdxSize;
use polars_utils::index::AtomicIdxSize;
use polars_utils::mem::prefetch::get_memory_prefetch_func;
use polars_utils::slice_enum::Slice;

use super::multi_file_reader::extra_ops::missing_columns::MissingColumnsPolicy;
use super::multi_file_reader::reader_interface::output::{
    FileReaderOutputRecv, FileReaderOutputSend,
};
use super::multi_file_reader::reader_interface::{
    BeginReadArgs, FileReader, FileReaderCallbacks, calc_row_position_after_slice,
};
use super::parquet::metadata_utils;
use crate::async_executor::{self};
use crate::nodes::compute_node_prelude::*;
use crate::nodes::io_sources::parquet::ParquetReadImpl;
use crate::nodes::{TaskPriority, io_sources};

pub struct ParquetFileReader {
    scan_source: ScanSource,
    cloud_options: Option<Arc<CloudOptions>>,
    config: Arc<ParquetOptions>,
    byte_source_builder: DynByteSourceBuilder,
    verbose: bool,

    /// Set during initialize()
    init_data: Option<InitializedState>,
}

struct InitializedState {
    file_metadata: Arc<FileMetadata>,
    file_schema: Arc<ArrowSchema>,
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

        let (metadata_bytes, opt_full_bytes) = {
            let byte_source = byte_source.clone();

            pl_async::get_runtime()
                .spawn(async move {
                    metadata_utils::read_parquet_metadata_bytes(&byte_source, verbose).await
                })
                .await
                .unwrap()?
        };

        let file_metadata = polars_parquet::parquet::read::deserialize_metadata(
            metadata_bytes.as_ref(),
            metadata_bytes.len() * 2 + 1024,
        )?;

        if let Some(full_bytes) = opt_full_bytes {
            byte_source = Arc::new(DynByteSource::MemSlice(MemSliceByteSource(full_bytes)));
        }

        let file_schema = infer_schema_with_options(&file_metadata, &None)?;

        let file_metadata = Arc::new(file_metadata);
        let file_schema = Arc::new(file_schema);

        self.init_data = Some(InitializedState {
            file_metadata,
            file_schema,
            byte_source,
        });

        Ok(())
    }

    fn begin_read(
        &self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)> {
        let verbose = self.verbose;

        let InitializedState {
            file_metadata,
            file_schema,
            byte_source,
        } = self.init_data.as_ref().unwrap();

        // TODO: Accept more stuff
        let BeginReadArgs {
            projected_schema,
            row_index,
            pre_slice: pre_slice_arg,
            mut predicate,
            cast_columns_policy: _,
            missing_columns_policy,
            num_pipelines,
            callbacks:
                FileReaderCallbacks {
                    file_schema_tx,
                    n_rows_in_file_tx,
                    row_position_on_end_tx,
                },
            scan_source_idx,
        } = args;

        let n_rows_in_file = self._n_rows_in_file()?;

        let normalized_pre_slice = pre_slice_arg
            .clone()
            .map(|x| x.restrict_to_bounds(usize::try_from(n_rows_in_file).unwrap()));

        // let file_schema_pl =
        //     std::sync::LazyLock::new(|| Arc::new(Schema::from_arrow_schema(file_schema.as_ref())));
        let file_schema_pl = Arc::new(Schema::from_arrow_schema(file_schema.as_ref()));

        // Parquet: We can send all callbacks immediately to unblock the next reader
        if let Some(file_schema_tx) = file_schema_tx {
            _ = file_schema_tx.send(file_schema_pl.clone());
        }

        if let Some(n_rows_in_file_tx) = n_rows_in_file_tx {
            _ = n_rows_in_file_tx.send(n_rows_in_file);
        }

        // We are allowed to send this value immediately, even though we haven't "ended" yet
        // (see its definition under FileReaderCallbacks).
        if let Some(row_position_on_end_tx) = row_position_on_end_tx {
            _ = row_position_on_end_tx
                .send(self._row_position_after_slice(normalized_pre_slice.clone())?);
            dbg!(scan_source_idx);
        }

        if normalized_pre_slice.as_ref().is_some_and(|x| x.len() == 0) {
            let (_, rx) = FileReaderOutputSend::new_serial();

            if verbose {
                eprintln!(
                    "[ParquetSource]: early return: \
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

        if let Some(predicate) = predicate.as_mut() {
            match missing_columns_policy {
                MissingColumnsPolicy::Raise => {
                    missing_columns_policy.initialize_policy(
                        projected_schema.as_ref(),
                        &file_schema_pl,
                        &mut vec![],
                    )?;
                },
                MissingColumnsPolicy::Insert => {
                    let v = projected_schema
                        .iter()
                        .filter(|(name, _)| !file_schema.contains(name))
                        .map(|(name, dtype)| {
                            (name.clone(), Scalar::new(dtype.clone(), AnyValue::Null))
                        })
                        .collect::<Vec<_>>();

                    if !v.is_empty() {
                        predicate.set_external_constant_columns(v);
                    }
                },
            }
        }

        let scan_source = self.scan_source.clone();

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

        // Note: The implementation we are dispatching to was made before we had multi
        // scan, hence the extensive materialization of config options.
        let (output_recv, handle) = ParquetReadImpl {
            scan_sources: scan_source.into_sources(),
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
            row_index: row_index.map(|ri| Arc::new((ri.name, AtomicIdxSize::new(ri.offset)))),
        }
        .run();

        Ok((
            output_recv,
            async_executor::spawn(TaskPriority::Low, async move { handle.await.unwrap() }),
        ))
    }

    async fn n_rows_in_file(&self) -> PolarsResult<IdxSize> {
        self._n_rows_in_file()
    }

    async fn row_position_after_slice(&self, pre_slice: Option<Slice>) -> PolarsResult<IdxSize> {
        self._row_position_after_slice(pre_slice)
    }
}

impl ParquetFileReader {
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
