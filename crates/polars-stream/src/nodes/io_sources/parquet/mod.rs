use std::sync::Arc;

use polars_core::config;
use polars_core::prelude::ArrowSchema;
use polars_core::schema::{Schema, SchemaExt, SchemaRef};
use polars_core::utils::arrow::bitmap::Bitmap;
use polars_core::utils::slice_offsets;
use polars_error::{PolarsResult, polars_err};
use polars_io::cloud::CloudOptions;
use polars_io::predicates::ScanIOPredicate;
use polars_io::prelude::{FileMetadata, ParquetOptions};
use polars_io::utils::byte_source::DynByteSourceBuilder;
use polars_io::{RowIndex, pl_async};
use polars_parquet::read::schema::infer_schema_with_options;
use polars_plan::dsl::{ScanSource, ScanSources};
use polars_plan::plans::FileInfo;
use polars_plan::prelude::FileScanOptions;
use polars_utils::IdxSize;
use polars_utils::index::AtomicIdxSize;
use polars_utils::mem::prefetch::get_memory_prefetch_func;
use polars_utils::pl_str::PlSmallStr;

use super::multi_scan::MultiScanable;
use super::{MorselOutput, RowRestriction, SourceNode, SourceOutput};
use crate::async_executor::spawn;
use crate::async_primitives::connector::{Receiver, connector};
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::SourceToken;
use crate::nodes::TaskPriority;
use crate::nodes::compute_node_prelude::*;
use crate::utils::task_handles_ext;

mod init;
mod metadata_utils;
mod row_group_data_fetch;
mod row_group_decode;

type AsyncTaskData = (
    Vec<crate::async_primitives::distributor_channel::Receiver<(DataFrame, MorselSeq)>>,
    task_handles_ext::AbortOnDropHandle<PolarsResult<()>>,
);

#[allow(clippy::type_complexity)]
pub struct ParquetSourceNode {
    scan_sources: ScanSources,
    file_info: FileInfo,
    predicate: Option<ScanIOPredicate>,
    options: ParquetOptions,
    cloud_options: Option<CloudOptions>,
    file_options: Box<FileScanOptions>,
    normalized_pre_slice: Option<(usize, usize)>,
    metadata: Arc<FileMetadata>,
    // Run-time vars
    config: Config,
    verbose: bool,
    schema: Option<Arc<ArrowSchema>>,
    projected_arrow_schema: Option<Arc<ArrowSchema>>,
    byte_source_builder: DynByteSourceBuilder,
    memory_prefetch_func: fn(&[u8]) -> (),
    /// The offset is an AtomicIdxSize, as in the negative slice case, the row
    /// offset becomes relative to the starting point in the list of files,
    /// so the row index offset needs to be updated by the initializer to
    /// reflect this (https://github.com/pola-rs/polars/issues/19607).
    row_index: Option<Arc<(PlSmallStr, AtomicIdxSize)>>,
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

#[allow(clippy::too_many_arguments)]
impl ParquetSourceNode {
    pub fn new(
        scan_sources: ScanSources,
        file_info: FileInfo,
        predicate: Option<ScanIOPredicate>,
        options: ParquetOptions,
        cloud_options: Option<CloudOptions>,
        mut file_options: Box<FileScanOptions>,
        metadata: Arc<FileMetadata>,
    ) -> Self {
        let verbose = config::verbose();

        let byte_source_builder = if scan_sources.is_cloud_url() || config::force_async() {
            DynByteSourceBuilder::ObjectStore
        } else {
            DynByteSourceBuilder::Mmap
        };
        let memory_prefetch_func = get_memory_prefetch_func(verbose);

        let row_index = file_options
            .row_index
            .take()
            .map(|ri| Arc::new((ri.name, AtomicIdxSize::new(ri.offset))));

        Self {
            scan_sources,
            file_info,
            predicate,
            options,
            cloud_options,
            file_options,
            normalized_pre_slice: None,
            metadata,

            config: Config {
                // Initialized later
                num_pipelines: 0,
                row_group_prefetch_size: 0,
                min_values_per_thread: 0,
            },
            verbose,
            schema: None,
            projected_arrow_schema: None,
            byte_source_builder,
            memory_prefetch_func,
            row_index,
        }
    }
}

impl SourceNode for ParquetSourceNode {
    fn name(&self) -> &str {
        "parquet_source"
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
        let (mut send_to, recv_from) = (0..state.num_pipelines)
            .map(|_| connector())
            .collect::<(Vec<_>, Vec<_>)>();

        self.config = {
            let row_group_prefetch_size = polars_core::config::get_rg_prefetch_size();

            // This can be set to 1 to force column-per-thread parallelism, e.g. for bug reproduction.
            let min_values_per_thread = std::env::var("POLARS_MIN_VALUES_PER_THREAD")
                .map(|x| x.parse::<usize>().expect("integer").max(1))
                .unwrap_or(16_777_216);

            Config {
                num_pipelines: state.num_pipelines,
                row_group_prefetch_size,
                min_values_per_thread,
            }
        };

        if self.verbose {
            eprintln!("[ParquetSource]: {:?}", &self.config);
        }

        self.normalized_pre_slice = self
            .file_options
            .pre_slice
            .map(|(offset, length)| slice_offsets(offset, length, self.metadata.num_rows));

        let num_rows = self.metadata.num_rows;
        self.schema = Some(self.file_info.reader_schema.take().unwrap().unwrap_left());
        self.init_projected_arrow_schema();

        let (raw_morsel_receivers, morsel_stream_task_handle) = self.init_raw_morsel_distributor();

        join_handles.push(spawn(TaskPriority::Low, async move {
            if let Some(rc) = unrestricted_row_count {
                let num_rows = IdxSize::try_from(num_rows)
                    .map_err(|_| polars_err!(bigidx, ctx = "parquet file", size = num_rows))?;
                _ = rc.send(num_rows);
            }

            // Every phase we are given a new send port.
            while let Ok(phase_output) = output_recv.recv().await {
                let source_token = SourceToken::new();
                let morsel_senders = phase_output.port.parallel();
                let mut morsel_outcomes = Vec::with_capacity(morsel_senders.len());
                for (send_to, port) in send_to.iter_mut().zip(morsel_senders) {
                    let (outcome, wait_group, morsel_output) =
                        MorselOutput::from_port(port, source_token.clone());
                    _ = send_to.send(morsel_output).await;
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

            // Join on the producer handle to catch errors/panics.
            // Safety
            // * We dropped the receivers on the line above
            // * This function is only called once.
            _ = morsel_stream_task_handle.await.unwrap();
            Ok(())
        }));

        join_handles.extend(recv_from.into_iter().zip(raw_morsel_receivers).map(
            |(mut recv_from, mut raw_morsel_rx)| {
                spawn(TaskPriority::Low, async move {
                    'port_recv: while let Ok(mut morsel_output) = recv_from.recv().await {
                        let wait_group = WaitGroup::default();

                        while let Ok((df, seq)) = raw_morsel_rx.recv().await {
                            let mut morsel =
                                Morsel::new(df, seq, morsel_output.source_token.clone());
                            morsel.set_consume_token(wait_group.token());
                            if morsel_output.port.send(morsel).await.is_err() {
                                break;
                            }

                            wait_group.wait().await;
                            if morsel_output.source_token.stop_requested() {
                                morsel_output.outcome.stop();
                                continue 'port_recv;
                            }
                        }

                        break;
                    }

                    Ok(())
                })
            },
        ));
    }
}

impl MultiScanable for ParquetSourceNode {
    type ReadOptions = ParquetOptions;

    const BASE_NAME: &'static str = "parquet";

    const SPECIALIZED_PRED_PD: bool = true;

    async fn new(
        source: ScanSource,
        options: &Self::ReadOptions,
        cloud_options: Option<&CloudOptions>,
        row_index: Option<PlSmallStr>,
    ) -> PolarsResult<Self> {
        let scan_sources = source.into_sources();

        let verbose = config::verbose();

        let scan_sources_2 = scan_sources.clone();
        let cloud_options_2 = cloud_options.cloned();

        // TODO: Use _opt_full_bytes if it is Some(_)
        let (metadata_bytes, _opt_full_bytes) = pl_async::get_runtime()
            .spawn(async move {
                let scan_sources = scan_sources_2;
                let cloud_options = cloud_options_2;
                let source = scan_sources.at(0);

                let byte_source = source
                    .to_dyn_byte_source(
                        &if scan_sources.is_cloud_url() || config::force_async() {
                            DynByteSourceBuilder::ObjectStore
                        } else {
                            DynByteSourceBuilder::Mmap
                        },
                        cloud_options.as_ref(),
                    )
                    .await?;

                metadata_utils::read_parquet_metadata_bytes(&byte_source, verbose).await
            })
            .await
            .unwrap()?;

        let file_metadata = polars_parquet::parquet::read::deserialize_metadata(
            metadata_bytes.as_ref(),
            metadata_bytes.len() * 2 + 1024,
        )?;

        let arrow_schema = infer_schema_with_options(&file_metadata, &None)?;
        let arrow_schema = Arc::new(arrow_schema);

        let schema = Schema::from_arrow_schema(&arrow_schema);
        let schema = Arc::new(schema);

        let mut options = options.clone();
        options.schema = Some(schema.clone());

        let file_options = Box::new(FileScanOptions {
            row_index: row_index.map(|name| RowIndex { name, offset: 0 }),
            ..Default::default()
        });

        let file_info = FileInfo::new(
            schema.clone(),
            Some(rayon::iter::Either::Left(arrow_schema.clone())),
            (None, usize::MAX),
        );

        Ok(ParquetSourceNode::new(
            scan_sources,
            file_info,
            None,
            options,
            cloud_options.cloned(),
            file_options,
            Arc::new(file_metadata),
        ))
    }

    fn with_projection(&mut self, projection: Option<&Bitmap>) {
        if let Some(projection) = projection {
            let mut with_columns = Vec::with_capacity(
                usize::from(self.file_options.row_index.is_some()) + projection.set_bits(),
            );

            if let Some(ri) = self.file_options.row_index.as_ref() {
                with_columns.push(ri.name.clone());
            }
            with_columns.extend(
                projection
                    .true_idx_iter()
                    .map(|idx| self.file_info.schema.get_at_index(idx).unwrap().0.clone())
                    .collect::<Vec<_>>(),
            );
            self.file_options.with_columns = Some(with_columns.into());
        }
    }
    fn with_row_restriction(&mut self, row_restriction: Option<RowRestriction>) {
        self.predicate = None;
        self.file_options.pre_slice = None;

        if let Some(row_restriction) = row_restriction {
            match row_restriction {
                RowRestriction::Slice(slice) => {
                    self.file_options.pre_slice = Some((slice.start as i64, slice.len()));
                },
                // @TODO: Cache
                RowRestriction::Predicate(scan_predicate) => {
                    self.predicate = Some(scan_predicate);
                },
            }
        }
    }

    async fn unrestricted_row_count(&mut self) -> PolarsResult<IdxSize> {
        let num_rows = self.metadata.num_rows;
        IdxSize::try_from(num_rows)
            .map_err(|_| polars_err!(bigidx, ctx = "parquet file", size = num_rows))
    }

    async fn physical_schema(&mut self) -> PolarsResult<SchemaRef> {
        Ok(self.file_info.schema.clone())
    }
}
