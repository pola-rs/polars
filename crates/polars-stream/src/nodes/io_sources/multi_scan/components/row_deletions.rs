use std::sync::{Arc, OnceLock, Weak};

#[cfg(feature = "python")]
use polars_arrow::array::ListArray;
use polars_arrow::array::{Array, BooleanArray};
use polars_arrow::bitmap::bitmask::BitMask;
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_async::executor::{self, AbortOnDropHandle, TaskMetricAggregator, TaskPriority};
use polars_buffer::Buffer;
use polars_core::frame::DataFrame;
use polars_core::prelude::{BooleanChunked, ChunkAgg, DataType, NamedFrom, PlIndexMap};
use polars_core::schema::{Schema, SchemaRef};
use polars_core::series::Series;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_error::{PolarsResult, feature_gated, polars_bail, polars_ensure, polars_err};
use polars_io::cloud::CloudOptions;
use polars_io::cloud::concurrency_config::FetchConfig;
use polars_io::utils::byte_source::{ByteSource, DynByteSourceBuilder};
#[cfg(feature = "python")]
use polars_plan::dsl::deletion::DeltaDeletionVectorProvider;
use polars_plan::dsl::deletion::{
    DELTA_DV_MAX_SERIALIZED_BYTES, DeletionFilesList, DeltaDeletionVector,
    DeltaDeletionVectorDecoder, DeltaDeletionVectorLocation, delta_deletion_vector_decoder,
    require_delta_deletion_vector_decoder,
};
use polars_plan::dsl::{CastColumnsPolicy, ScanSource, ScanSources};
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_path::PlRefPath;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::slice_enum::Slice;

use crate::metrics::IOMetrics;
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::{BeginReadArgs, FileReaderCallbacks};
#[cfg(feature = "parquet")]
use crate::nodes::io_sources::parquet::builder::ParquetReaderBuilder;

#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum DeletionFilesProvider {
    None,

    #[cfg(feature = "parquet")]
    Iceberg {
        data_paths: Buffer<PlRefPath>,
        paths: Arc<PlIndexMap<usize, polars_plan::dsl::deletion::IcebergDeletes>>,
        // Amortized allocations
        reader_builder: Box<ParquetReaderBuilder>,
        projected_schema: SchemaRef,
        deletion_load_tasks: Arc<
            parking_lot::Mutex<
                PlHashMap<PlRefPath, Arc<tokio::sync::OnceCell<PuffinDeletionVectors>>>,
            >,
        >,
    },
    DeltaNative {
        vectors: Arc<PlIndexMap<usize, DeltaDeletionVector>>,
        cache: Arc<parking_lot::Mutex<NativeDeletionVectorCache>>,
        files: Arc<DeletionVectorFiles>,
        task_metrics: Option<Arc<TaskMetricAggregator>>,
        io_metrics: Option<Arc<IOMetrics>>,
    },
    #[cfg(feature = "python")]
    DeltaDeletionVector {
        provider: DeltaDeletionVectorProvider,
        selected_paths: Buffer<PlRefPath>,
        cache: Arc<tokio::sync::OnceCell<Option<ListArray<i64>>>>,
        task_metrics: Option<Arc<TaskMetricAggregator>>,
    },
}

impl DeletionFilesProvider {
    pub fn try_new(
        deletion_files: Option<DeletionFilesList>,
        selected_sources: ScanSources,
        execution_state: &crate::execute::StreamingExecutionState,
        io_metrics: Option<Arc<IOMetrics>>,
    ) -> PolarsResult<Self> {
        match deletion_files {
            Some(DeletionFilesList::Iceberg(paths)) => feature_gated!("parquet", {
                let mut reader_builder = ParquetReaderBuilder {
                    bytes_per_source: None,
                    first_metadata: None,
                    field_selection: None,
                    scan_schema: None,
                    prune_to_scan_schema: false,
                    options: Arc::new(polars_io::prelude::ParquetOptions {
                        schema: Some(Arc::new(Schema::from_iter([
                            (PlSmallStr::from_static("file_path"), DataType::String),
                            (PlSmallStr::from_static("pos"), DataType::Int64),
                        ]))),

                        parallel: polars_io::prelude::ParallelStrategy::Auto,
                        low_memory: false,
                        use_statistics: false,
                        decryption: None,
                    }),
                    pipeline_budget: std::sync::OnceLock::new(),
                    shared_prefetch_wait_group_slot: Default::default(),
                    file_read_context: std::sync::OnceLock::new(),
                    io_metrics: io_metrics.map(OnceLock::from).unwrap_or_default(),
                    task_metrics: std::sync::OnceLock::new(),
                };

                reader_builder.set_execution_state(execution_state);
                // Deletion files keep a fixed byte window: no growth, no pool draw.
                let pipeline_budget = reader_builder.pipeline_budget.take().unwrap();
                reader_builder
                    .pipeline_budget
                    .set(pipeline_budget.with_fixed_kbytes())
                    .unwrap();

                let ScanSources::Paths(data_paths) = selected_sources else {
                    polars_bail!(
                        ComputeError:
                        "cannot use iceberg deletion files with non-paths sources"
                    )
                };

                Ok(Self::Iceberg {
                    data_paths,
                    paths,
                    reader_builder: Box::new(reader_builder),
                    projected_schema: Arc::new(Schema::from_iter([
                        (PlSmallStr::from_static("file_path"), DataType::String),
                        (PlSmallStr::from_static("pos"), DataType::Int64),
                    ])),
                    deletion_load_tasks: Default::default(),
                })
            }),
            Some(DeletionFilesList::Delta(vectors)) => Ok(Self::DeltaNative {
                files: Arc::new(DeletionVectorFiles::new(&vectors)),
                vectors,
                cache: Default::default(),
                task_metrics: execution_state.task_metrics.clone(),
                io_metrics,
            }),
            #[cfg(feature = "python")]
            Some(DeletionFilesList::DeltaPythonCallback(provider)) => {
                let ScanSources::Paths(selected_paths) = selected_sources else {
                    polars_bail!(ComputeError: "delta deletion vectors require path-based scan sources");
                };

                Ok(Self::DeltaDeletionVector {
                    provider,
                    selected_paths,
                    cache: Arc::new(tokio::sync::OnceCell::new()),
                    task_metrics: execution_state.task_metrics.clone(),
                })
            },
            None => Ok(Self::None),
        }
    }

    pub fn spawn_row_deletions_init(
        &self,
        scan_source_idx: usize,
        cloud_options: Option<Arc<CloudOptions>>,
        num_pipelines: usize,
        verbose: bool,
    ) -> Option<RowDeletionsInit> {
        match self {
            Self::None => None,

            #[cfg(feature = "parquet")]
            Self::Iceberg {
                data_paths,
                paths,
                reader_builder,
                projected_schema,
                deletion_load_tasks,
            } => {
                use std::pin::Pin;

                use polars_plan::dsl::deletion::IcebergDeletes;
                use polars_plan::dsl::{ExtraColumnsPolicy, MissingColumnsPolicy};

                let paths = paths.get(&scan_source_idx)?;
                let task_metrics = reader_builder.task_metrics.get().map(Arc::as_ref);

                if verbose {
                    let s = if paths.len() == 1 { "" } else { "s" };
                    eprintln!(
                        "[DeletionFilesProvider[Iceberg]]: scan_source_idx: {}, {} file{s}",
                        scan_source_idx,
                        paths.len(),
                    )
                }

                let deletion_files_init_fut: Pin<
                    Box<dyn Future<Output = PolarsResult<ExternalFilterMask>> + Send>,
                > = match paths {
                    IcebergDeletes::PositionDeletes(paths) => {
                        let file_readers = paths
                            .iter()
                            .enumerate()
                            .map(|(deletion_file_idx, path)| {
                                let source = ScanSource::Path(path.clone());
                                let reader = reader_builder.build_file_reader(
                                    source,
                                    cloud_options.clone(),
                                    deletion_file_idx,
                                );

                                if verbose {
                                    eprintln!(
                                        "[DeletionFilesProvider[Iceberg]]: \
                                        scan_source_idx: {scan_source_idx}, \
                                        deletion_file_idx: {deletion_file_idx}, \
                                        deletion_file_path: {path}, \
                                        deletion_file_type: position-delete"
                                    )
                                }

                                AbortOnDropHandle::new(executor::spawn(
                                    TaskPriority::Low,
                                    task_metrics,
                                    async move {
                                        let mut reader = reader?;
                                        reader.initialize().await?;
                                        PolarsResult::Ok(reader)
                                    },
                                ))
                            })
                            .collect::<Vec<_>>();

                        let projected_schema = projected_schema.clone();
                        let read_task_metrics = reader_builder.task_metrics.get().cloned();

                        // We choose to load deletion files immediately during the initialization phase -
                        // the main driver loop of the multi file may need to serially `.await` on this
                        // between initializing readers when there is a slice.
                        //
                        // This does mean deletion file loads are tied to `NUM_READERS_PRE_INIT`, but this
                        // should be fine as the size of the data should not be too big.

                        Box::pin(async move {
                            let mut handles = Vec::with_capacity(file_readers.len());

                            for init_fut in file_readers {
                                let mut reader = init_fut.await?;

                                use crate::nodes::io_sources::multi_scan::components::projection::Projection;

                                let begin_read_args = BeginReadArgs {
                                    projection: Projection::Plain(projected_schema.clone()),
                                    row_index: None,
                                    pre_slice: None,
                                    predicate: None,
                                    cast_columns_policy: CastColumnsPolicy::ERROR_ON_MISMATCH,
                                    extra_columns_policy: ExtraColumnsPolicy::Raise,
                                    missing_columns_policy: MissingColumnsPolicy::Raise,
                                    num_pipelines,
                                    disable_morsel_split: false,
                                    maintain_order: true,
                                    last_morsel_pipelines: 1,
                                    callbacks: FileReaderCallbacks {
                                        file_schema_tx: None,
                                        n_rows_in_file_tx: None,
                                        row_position_on_end_tx: None,
                                    },
                                };

                                let projected_schema = projected_schema.clone();

                                handles.push(AbortOnDropHandle::new(executor::spawn(
                                    TaskPriority::Low,
                                    read_task_metrics.as_deref(),
                                    async move {
                                        let (mut rx, handle) =
                                            reader.begin_read(begin_read_args)?;

                                        let mut dfs = vec![];

                                        while let Ok(morsel) = rx.recv().await {
                                            dfs.push(morsel.into_df().await);
                                        }

                                        handle.await?;

                                        // An empty file sends no morsels.
                                        if dfs.is_empty() {
                                            dfs.push(DataFrame::empty_with_schema(
                                                &projected_schema,
                                            ));
                                        }

                                        let df = accumulate_dataframes_vertical_unchecked(dfs);

                                        // Some quick testing on AWS Athena showed that it doesn't
                                        // write deletion files that reference multiple distinct
                                        // file paths, so we don't handle that for now.
                                        assert!(
                                            df.column("file_path")?.n_unique()? <= 1,
                                            "assertion failed: iceberg position delete file: \
                                            n_unique(data_file_paths) <= 1. \
                                            This is a bug, please open an issue"
                                        );

                                        let positions_col = df.column("pos")?.clone();
                                        let max_idx = usize::try_from(
                                            positions_col
                                                .as_materialized_series_maintain_scalar()
                                                .i64()
                                                .unwrap()
                                                .max()
                                                .unwrap_or(0),
                                        )
                                        .unwrap();

                                        PolarsResult::Ok((positions_col, max_idx))
                                    },
                                )))
                            }

                            let mut position_columns = Vec::with_capacity(handles.len());
                            let mut filter_mask_len: usize = 0;

                            for handle in handles {
                                let (positions_col, max_idx) = handle.await?;
                                filter_mask_len = filter_mask_len.max(max_idx.saturating_add(1));
                                position_columns.push(positions_col);
                            }

                            let mut filter_mask = MutableBitmap::from_len_set(filter_mask_len);

                            for c in position_columns {
                                for idx in c
                                    .as_materialized_series_maintain_scalar()
                                    .i64()
                                    .unwrap()
                                    .iter()
                                {
                                    let idx = usize::try_from(idx.unwrap()).unwrap();
                                    filter_mask.set(idx, false);
                                }
                            }

                            let bitmap = filter_mask.freeze();

                            // Also trigger the bitcount to reduce blocking later down.
                            bitmap.unset_bits();
                            debug_assert!(bitmap.lazy_unset_bits().is_some());

                            let mask = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, bitmap);
                            let mask = ExternalFilterMask::Iceberg { mask };

                            Ok(mask)
                        })
                    },

                    IcebergDeletes::DeletionVector {
                        path,
                        content_offset,
                    } => {
                        use polars_async::ASYNC;
                        use polars_utils::async_utils::tokio_handle_ext;

                        let deletion_load_tasks = Arc::clone(deletion_load_tasks);
                        let data_path = data_paths[scan_source_idx].clone();
                        let path = path.clone();
                        let content_offset = *content_offset;
                        let cloud_options = cloud_options.clone();
                        let io_metrics = reader_builder.io_metrics.get().cloned();

                        Box::pin(async move {
                            let deletions = {
                                let mut tasks = deletion_load_tasks.lock();
                                Arc::clone(tasks.entry(path.clone()).or_default())
                            };
                            let dv_path = path.clone();

                            deletions
                                .get_or_try_init(|| async move {
                                    let fetch_bytes_handle =
                                        tokio_handle_ext::AbortOnDropHandle(ASYNC.spawn({
                                            async move {
                                                let byte_source =
                                                    DynByteSourceBuilder::ObjectStore(
                                                        FetchConfig::streaming(),
                                                    )
                                                    .try_build_from_path(
                                                        path,
                                                        cloud_options.as_deref(),
                                                        io_metrics,
                                                    )
                                                    .await?;
                                                let len = byte_source.get_size().await?;

                                                byte_source.get_range(0..len).await
                                            }
                                        }));

                                    let puffin_bytes = fetch_bytes_handle.await.unwrap()?;
                                    load_iceberg_puffin_deletes(puffin_bytes)
                                })
                                .await?
                                .get(&dv_path, &data_path, content_offset)
                        })
                    },
                };

                let handle = AbortOnDropHandle::new(executor::spawn(
                    TaskPriority::Low,
                    task_metrics,
                    async move {
                        let mask = deletion_files_init_fut.await?;

                        if verbose {
                            let num_deleted_rows = mask.num_deleted_rows(mask.len());
                            let max_index = mask.len().checked_sub(1);

                            eprintln!(
                                "[DeletionFilesProvider[Iceberg]]: \
                                scan_source_idx: {scan_source_idx}, \
                                num_deleted_rows: {num_deleted_rows}, \
                                max_index: {max_index:?}",
                            )
                        }

                        Ok(mask)
                    },
                ));

                Some(RowDeletionsInit::Initializing(handle))
            },

            Self::DeltaNative {
                vectors,
                cache,
                files,
                task_metrics,
                io_metrics,
            } => {
                let files = files.clone();
                let vector = vectors.get(&scan_source_idx)?.clone();
                let cell = native_cache_cell(cache, &vector);
                let io_metrics = io_metrics.clone();
                if verbose {
                    match &vector.location {
                        DeltaDeletionVectorLocation::Inline(_) => eprintln!(
                            "[DeletionFilesProvider[Delta]]: scan_source_idx: {scan_source_idx}, inline"
                        ),
                        DeltaDeletionVectorLocation::File { path, offset } => eprintln!(
                            "[DeletionFilesProvider[Delta]]: scan_source_idx: {scan_source_idx}, path: {path}, offset: {offset}"
                        ),
                    }
                }
                let handle = AbortOnDropHandle::new(executor::spawn(
                    TaskPriority::Low,
                    task_metrics.as_deref(),
                    async move {
                        cell.get_or_try_init(|| async move {
                            let decoder = match delta_deletion_vector_decoder() {
                                Some(d) => d,
                                // Loading takes the Python GIL: not on an executor thread.
                                // Scans starting together load once, the rest wait for it
                                // (an async lock holds neither a thread nor the GIL).
                                None => {
                                    static LOADING: tokio::sync::Mutex<()> =
                                        tokio::sync::Mutex::const_new(());
                                    let _loading = LOADING.lock().await;
                                    match delta_deletion_vector_decoder() {
                                        Some(d) => d,
                                        None => polars_core::runtime::ASYNC
                                            .spawn_blocking(require_delta_deletion_vector_decoder)
                                            .await
                                            .unwrap()?,
                                    }
                                },
                            };
                            let max_rows = deletion_vector_max_rows();
                            let bitmap = match vector.location {
                                DeltaDeletionVectorLocation::Inline(bytes) => decoder
                                    .decode_inline(
                                        &bytes,
                                        vector.size_in_bytes,
                                        vector.cardinality,
                                        max_rows,
                                    )?,
                                DeltaDeletionVectorLocation::File { path, offset } => {
                                    let bytes = files
                                        .read_frame(
                                            decoder.clone(),
                                            path,
                                            offset,
                                            vector.size_in_bytes,
                                            cloud_options.as_deref(),
                                            io_metrics,
                                        )
                                        .await?;
                                    decoder.decode_frame(
                                        &bytes,
                                        offset,
                                        vector.size_in_bytes,
                                        vector.cardinality,
                                        max_rows,
                                    )?
                                },
                            };
                            bitmap.unset_bits();
                            Ok(ExternalFilterMask::DeltaDeletionVector {
                                mask: BooleanChunked::from_bitmap(PlSmallStr::EMPTY, bitmap),
                            })
                        })
                        .await
                        .cloned()
                    },
                ));
                Some(RowDeletionsInit::Initializing(handle))
            },

            #[cfg(feature = "python")]
            Self::DeltaDeletionVector {
                provider,
                selected_paths,
                cache,
                task_metrics,
            } => {
                let cache = cache.clone();
                let provider = provider.clone();
                let selected_paths = selected_paths.clone();

                let handle = AbortOnDropHandle::new(executor::spawn(
                    TaskPriority::Low,
                    task_metrics.as_deref(),
                    async move {
                        let deletion_vectors = cache
                            .get_or_try_init(|| async {
                                let provider = provider.clone();
                                let selected_paths = selected_paths.clone();
                                polars_core::runtime::ASYNC
                                    .spawn_blocking(move || provider.call(selected_paths))
                                    .await
                                    .unwrap()
                            })
                            .await?;

                        let empty_mask = BooleanChunked::new(PlSmallStr::EMPTY, [] as [bool; 0]);

                        let mask = match deletion_vectors {
                            None => empty_mask,
                            Some(list) if list.is_null(scan_source_idx) => empty_mask,
                            Some(list) => {
                                let arr = list.value(scan_source_idx);
                                let bool_arr = arr
                                    .as_any()
                                    .downcast_ref::<BooleanArray>()
                                    .ok_or_else(|| {
                                        polars_err!(ComputeError:
                                            "expected boolean array in Delta deletion vector")
                                    })?;
                                unsafe {
                                    BooleanChunked::from_chunks(
                                        PlSmallStr::EMPTY,
                                        vec![Box::new(bool_arr.clone())],
                                    )
                                }
                            },
                        };

                        Ok(ExternalFilterMask::DeltaDeletionVector { mask })
                    },
                ));

                Some(RowDeletionsInit::Initializing(handle))
            },
        }
    }
}

pub enum RowDeletionsInit {
    Initializing(AbortOnDropHandle<PolarsResult<ExternalFilterMask>>),

    /// If negative slice resolution happens we will already have the initialized filter mask
    /// much earlier in the pipeline, but the channel interface still requires a `RowDeletionsInit`,
    /// so this enum variant is used to hold the initialized mask.
    Initialized(ExternalFilterMask),
}

impl RowDeletionsInit {
    /// Loads the deletion information into a filter mask.
    pub async fn into_external_filter_mask(self) -> PolarsResult<ExternalFilterMask> {
        match self {
            Self::Initializing(handle) => handle.await,
            Self::Initialized(v) => Ok(v),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ExternalFilterMask {
    /// Note: Iceberg positional deletes can have a mask length shorter than the actual data.
    Iceberg { mask: BooleanChunked },
    /// Delta deletion vector.
    /// Note: technically this is a selection vector, i.e. true = keep, false = drop.
    DeltaDeletionVector { mask: BooleanChunked },
}

impl Default for ExternalFilterMask {
    fn default() -> Self {
        Self::Iceberg {
            mask: BooleanChunked::default(),
        }
    }
}

impl ExternalFilterMask {
    pub fn variant_name(&self) -> &'static str {
        use ExternalFilterMask::*;
        match self {
            Iceberg { .. } => "Iceberg",
            DeltaDeletionVector { .. } => "DeltaDeletionVector",
        }
    }

    /// Human-friendly verbose log display.
    pub fn log_display(this: Option<&Self>) -> PlSmallStr {
        match this {
            None => PlSmallStr::from_static("None"),
            Some(mask) => {
                let mask_variant = mask.variant_name();
                let n = mask.num_deleted_rows(mask.len());
                let s = if n == 1 { "" } else { "s" };
                format_pl_smallstr!("{mask_variant}(<{n} deletion{s}>)")
            },
        }
    }

    pub fn filter_df(&self, df: &mut DataFrame) -> PolarsResult<()> {
        match self {
            Self::Iceberg { mask } => {
                if !mask.is_empty() {
                    *df = if mask.len() < df.height() {
                        accumulate_dataframes_vertical_unchecked([
                            df.slice(Slice::head(mask.len())).filter_seq(mask)?,
                            df.slice((mask.len(), df.height() - mask.len())),
                        ])
                    } else {
                        df.filter_seq(mask)?
                    }
                }
            },
            Self::DeltaDeletionVector { mask } => {
                if !mask.is_empty() {
                    *df = if mask.len() < df.height() {
                        accumulate_dataframes_vertical_unchecked([
                            df.slice(Slice::head(mask.len())).filter_seq(mask)?,
                            df.slice((mask.len(), df.height() - mask.len())),
                        ])
                    } else {
                        df.filter_seq(mask)?
                    }
                }
            },
        }

        Ok(())
    }

    pub fn slice(&self, offset: usize, len: usize) -> Self {
        match self {
            Self::Iceberg { mask } => {
                // This is not a valid offset, it's also a sentinel value from `RowCounter::MAX`.
                assert_ne!(offset, usize::MAX);
                let offset = offset.min(mask.len());
                let len = len.min(mask.len() - offset);

                let mask = mask.slice((offset, len));

                Self::Iceberg { mask }
            },
            Self::DeltaDeletionVector { mask } => {
                // This is not a valid offset, it's also a sentinel value from `RowCounter::MAX`.
                assert_ne!(offset, usize::MAX);
                let offset = offset.min(mask.len());
                let len = len.min(mask.len() - offset);

                let mask = mask.slice((offset, len));

                Self::DeltaDeletionVector { mask }
            },
        }
    }

    /// Number of deleted rows within the first `num_physical_rows` rows.
    ///
    /// Iceberg position deletes can reference positions past the end of the data file, these must
    /// not be counted as deleted rows.
    pub fn num_deleted_rows(&self, num_physical_rows: usize) -> usize {
        let mask = self.get_mask();

        if num_physical_rows >= mask.len() {
            mask.unset_bits()
        } else {
            mask.sliced(0, num_physical_rows).unset_bits()
        }
    }

    /// Calculates the physical pre_slice that can be applied before performing row deletions.
    ///
    /// By default, a `pre_slice` is applied after rows are deleted. This function takes a `pre_slice`
    /// and translates it to `physical` positions (i.e. one that can be applied before row deletions).
    /// This is done by expanding the range of the slice to account for the deleted rows.
    ///
    /// This involves 2 `nth_set_bit` searches for offset and length.
    ///
    /// # Panics
    /// Panics if `slice` is negative.
    pub fn calc_physical_slice(&self, slice: Slice) -> Slice {
        let mask = self.get_mask();

        let phys_slice = match slice {
            Slice::Positive { offset, len } => {
                let phys_offset = nth_set_bit_extend(&mask, offset);

                let phys_len = if phys_offset >= mask.len() || len == 0 {
                    // We are past any row deletions
                    len
                } else {
                    let mask = mask.clone().sliced(phys_offset, mask.len() - phys_offset);
                    nth_set_bit_extend(&mask, len - 1).saturating_add(1)
                };

                Slice::Positive {
                    offset: phys_offset,
                    len: phys_len,
                }
            },

            Slice::PositiveUnbounded { offset } => {
                Slice::skip(nth_set_bit_extend(&mask, offset.get())).unwrap()
            },

            // We cannot calculate the physical slice for the negative case because we don't know
            // the total length of the file.
            //
            // TODO: Bitmap-based deletion vectors (e.g. Delta, Iceberg V3) are sized as the exact
            // row-count, so the slice can be normalized to positive before reaching here.
            Slice::Negative { .. } => {
                panic!()
            },
        };

        assert!(phys_slice.len() >= slice.len());

        phys_slice
    }

    fn get_mask(&self) -> Bitmap {
        match self {
            Self::Iceberg { mask } => mask.rechunk().downcast_get(0).unwrap().values().clone(),
            Self::DeltaDeletionVector { mask } => {
                mask.rechunk().downcast_get(0).unwrap().values().clone()
            },
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Iceberg { mask } => mask.len(),
            Self::DeltaDeletionVector { mask } => mask.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Calculates the nth set bit as though `mask` were extended infinitely with trues.
fn nth_set_bit_extend(mask: &Bitmap, n: usize) -> usize {
    if let Some(n_additional) = n.checked_sub(mask.set_bits()) {
        mask.len().saturating_add(n_additional)
    } else {
        BitMask::from_bitmap(mask).nth_set_bit_idx(n, 0).unwrap()
    }
}

/// `file://C:/x` → `file:///C:/x`, as data file paths are normalized when the scan is
/// resolved, so that deletion vectors are found by their referenced data file.
#[cfg(feature = "python")]
fn normalize_windows_file_uri(path: &str) -> std::borrow::Cow<'_, str> {
    match path.strip_prefix("file://") {
        Some(rest) if matches!(rest.as_bytes(), [d, b':', ..] if d.is_ascii_alphabetic()) => {
            format!("file:///{rest}").into()
        },
        _ => path.into(),
    }
}

/// The deletion vectors of a Puffin file.
pub struct PuffinDeletionVectors {
    /// `(referenced data file, mask)` by blob offset.
    by_offset: PlHashMap<u64, (PlRefPath, ExternalFilterMask)>,
    /// Blob offsets by referenced data file.
    offsets_by_data_path: PlHashMap<PlRefPath, Vec<u64>>,
}

impl PuffinDeletionVectors {
    fn get(
        &self,
        dv_path: &PlRefPath,
        data_path: &PlRefPath,
        content_offset: Option<u64>,
    ) -> PolarsResult<ExternalFilterMask> {
        let offset = match content_offset {
            Some(offset) => offset,
            None => match self.offsets_by_data_path.get(data_path).map(Vec::as_slice) {
                Some(&[offset]) => offset,
                Some([]) | None => polars_bail!(
                    ComputeError:
                    "iceberg: deletion vector file {} has no deletion vector for data file {}",
                    dv_path, data_path
                ),
                Some(_) => polars_bail!(
                    ComputeError:
                    "iceberg: deletion vector file {} has several deletion vectors for data \
                    file {}, and the delete file entry has no content offset",
                    dv_path, data_path
                ),
            },
        };

        match self.by_offset.get(&offset) {
            Some((referenced, mask)) if referenced == data_path => Ok(mask.clone()),
            Some((referenced, _)) => polars_bail!(
                ComputeError:
                "iceberg: deletion vector at offset {} of {} is of data file {}, expected {}",
                offset, dv_path, referenced, data_path
            ),
            None => polars_bail!(
                ComputeError:
                "iceberg: deletion vector file {} has no deletion vector at offset {}",
                dv_path, offset
            ),
        }
    }
}

fn load_iceberg_puffin_deletes(puffin_bytes: Buffer<u8>) -> PolarsResult<PuffinDeletionVectors> {
    #[cfg(feature = "python")]
    {
        use std::sync::LazyLock;

        use polars_arrow::array::UInt64Array;
        use polars_error::constants::LENGTH_LIMIT_MSG;
        use polars_error::polars_ensure;
        use polars_utils::index::idxsize_try_from;
        use pyo3::{Py, PyAny, Python};

        let deletions: Vec<(u64, PlRefPath, Series)> = Python::attach(|py| {
            use pyo3::pybacked::PyBackedStr;
            use pyo3::types::{PyAnyMethods, PyTuple};

            if LazyLock::get(&LOAD_FROM_BYTES_FN).is_none() {
                py.detach(|| LazyLock::force(&LOAD_FROM_BYTES_FN));
            }

            // `[(offset, referenced data file, positions)]`
            let blobs = LOAD_FROM_BYTES_FN.call1(py, (puffin_bytes.as_slice(),))?;
            let blobs = blobs.bind(py);

            blobs
                .try_iter()?
                .map(|blob| {
                    let blob = blob?;
                    let blob = blob.cast::<PyTuple>()?;
                    Ok((
                        blob.get_item(0)?.extract::<u64>()?,
                        PlRefPath::new(&*normalize_windows_file_uri(
                            &blob.get_item(1)?.extract::<PyBackedStr>()?,
                        )),
                        *(polars_utils::python_convert_registry::get_python_convert_registry()
                            .from_py
                            .series)(
                            blob.get_item(2)?.getattr("_s")?.into()
                        )
                        .unwrap()
                        .downcast::<Series>()
                        .unwrap(),
                    ))
                })
                .collect::<PolarsResult<Vec<(u64, PlRefPath, Series)>>>()
        })?;

        let mut out = PuffinDeletionVectors {
            by_offset: PlHashMap::with_capacity(deletions.len()),
            offsets_by_data_path: PlHashMap::with_capacity(deletions.len()),
        };

        for (offset, data_path, indices) in deletions {
            let filter_mask_len = indices.max::<u64>()?.map_or(0, |x| x.saturating_add(1));

            polars_ensure!(
                idxsize_try_from(filter_mask_len).is_ok(),
                ComputeError: LENGTH_LIMIT_MSG
            );

            let indices = indices.rechunk();
            let indices: &UInt64Array = indices.chunks()[0].as_any().downcast_ref().unwrap();

            let mut filter_mask = MutableBitmap::from_len_set(filter_mask_len as usize);

            for idx in indices.non_null_values_iter() {
                let idx = usize::try_from(idx).unwrap();
                filter_mask.set(idx, false);
            }

            let bitmap = filter_mask.freeze();

            // Also trigger the bitcount to reduce blocking later down.
            bitmap.unset_bits();

            let mask = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, bitmap);
            let mask = ExternalFilterMask::Iceberg { mask };

            out.offsets_by_data_path
                .entry(data_path.clone())
                .or_default()
                .push(offset);
            out.by_offset.insert(offset, (data_path, mask));
        }

        return Ok(out);

        static LOAD_FROM_BYTES_FN: LazyLock<Py<PyAny>> = LazyLock::new(|| {
            Python::attach(|py| {
                use pyo3::types::PyAnyMethods;

                py.import("polars.io.iceberg._utils")
                    .unwrap()
                    .getattr("load_puffin_deletion_file")
                    .unwrap()
                    .unbind()
            })
        });
    }
    #[cfg(not(feature = "python"))]
    {
        panic!("failed to load iceberg deletion vector: 'python' feature not enabled")
    }
}

// Keep corrupt or highly sparse vectors from driving an unbounded dense allocation.
// This is a PER-MASK execution resource limit, also on bigidx builds, not an
// aggregate budget: the bounded active-reader window can hold several such masks. The decoder accepts
// a row bound so a future reader can instead pass the data file's validated row count.
fn deletion_vector_max_rows() -> usize {
    ((DELTA_DV_MAX_SERIALIZED_BYTES as u128 * 8)
        .min(u128::from(polars_core::prelude::IdxSize::MAX))) as usize
}

/// Vectors of a file are read together as one `[0, end)` request when that is at most this large.
const MAX_SHARED_PREFIX_BYTES: u64 = 1 << 20;
/// Prefixes held by a scan, over all files. Past it, files read their vectors one by one.
const SHARED_PREFIX_BUDGET_BYTES: u64 = 64 << 20;

struct DeletionVectorFileState {
    /// The format version byte of this file was validated.
    version_ok: tokio::sync::OnceCell<()>,
    /// `[0, span_end)` of the file, shared by its vectors, kept for the scan. `None`: not
    /// shared (over the budget, or the read failed), and each vector reads its own frame.
    prefix: tokio::sync::OnceCell<Option<Buffer<u8>>>,
    /// The file length, once a shared prefix read came back short (`u64::MAX`: unknown).
    known_len: std::sync::atomic::AtomicU64,
}

impl Default for DeletionVectorFileState {
    fn default() -> Self {
        Self {
            version_ok: Default::default(),
            prefix: Default::default(),
            known_len: u64::MAX.into(),
        }
    }
}

/// Bytes taken from the shared-prefix budget, returned on drop (e.g. when the reading task is
/// cancelled) unless [`commit`](Self::commit)ted.
struct PrefixReservation<'a> {
    files: &'a DeletionVectorFiles,
    n: u64,
    committed: bool,
}

impl PrefixReservation<'_> {
    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for PrefixReservation<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.files
                .prefix_budget
                .fetch_add(self.n, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Per-scan state of the files that hold deletion vectors.
pub struct DeletionVectorFiles {
    /// Largest end offset of the vectors of each file.
    span_end: PlHashMap<PlRefPath, u64>,
    files: parking_lot::Mutex<PlHashMap<PlRefPath, Arc<DeletionVectorFileState>>>,
    /// Bytes of shared prefixes that may still be read.
    prefix_budget: std::sync::atomic::AtomicU64,
}

impl DeletionVectorFiles {
    fn new(vectors: &PlIndexMap<usize, DeltaDeletionVector>) -> Self {
        let mut span_end: PlHashMap<PlRefPath, u64> = PlHashMap::new();
        for v in vectors.values() {
            if let DeltaDeletionVectorLocation::File { path, offset } = &v.location {
                let end = offset.saturating_add(u64::from(v.size_in_bytes) + 8);
                let e = span_end.entry(path.clone()).or_default();
                *e = (*e).max(end);
            }
        }
        Self {
            span_end,
            files: Default::default(),
            prefix_budget: SHARED_PREFIX_BUDGET_BYTES.into(),
        }
    }

    fn try_reserve(&self, n: u64) -> Option<PrefixReservation<'_>> {
        use std::sync::atomic::Ordering::Relaxed;
        self.prefix_budget
            .try_update(Relaxed, Relaxed, |b| b.checked_sub(n))
            .ok()
            .map(|_| PrefixReservation {
                files: self,
                n,
                committed: false,
            })
    }

    /// Read the frame of one vector: `size + 8` bytes at `offset`, after validating the file's
    /// version byte. A range past the end of the file is a short-read `ComputeError` whenever
    /// the file length is known from a short body (never an object-store range error then).
    async fn read_frame(
        self: &Arc<Self>,
        decoder: Arc<dyn DeltaDeletionVectorDecoder>,
        path: PlRefPath,
        offset: u64,
        size: u32,
        cloud_options: Option<&CloudOptions>,
        io_metrics: Option<Arc<IOMetrics>>,
    ) -> PolarsResult<Buffer<u8>> {
        polars_ensure!(
            size as usize <= DELTA_DV_MAX_SERIALIZED_BYTES,
            ComputeError: "deletion vector exceeds 512 MiB serialized size resource limit"
        );
        let range = usize::try_from(offset)
            .ok()
            .and_then(|start| Some(start..start.checked_add(size as usize)?.checked_add(8)?))
            .ok_or_else(
                || polars_err!(ComputeError: "deletion vector frame outside file bounds"),
            )?;
        let state = self.files.lock().entry(path.clone()).or_default().clone();
        let span_end = self.span_end.get(&path).copied().unwrap_or(0);
        let this = self.clone();
        let cloud_options = cloud_options.cloned();
        let source_path = path.clone();
        let handle = polars_utils::async_utils::tokio_handle_ext::AbortOnDropHandle(
            polars_core::runtime::ASYNC.spawn(async move {
                let source = DynByteSourceBuilder::ObjectStore(FetchConfig::streaming())
                    .try_build_from_path(source_path.clone(), cloud_options.as_ref(), io_metrics)
                    .await?;
                // A range past the end of the file is an error, whatever the store does with it
                // (a short body or a failure).
                let get = |r: std::ops::Range<usize>| {
                    let source = &source;
                    let source_path = &source_path;
                    let state = &state;
                    async move {
                        let b = source.get_range(r.clone()).await?;
                        if b.len() != r.len() {
                            if r.start == 0 {
                                state
                                    .known_len
                                    .store(b.len() as u64, std::sync::atomic::Ordering::Relaxed);
                            }
                            return Err(short_read_err(source_path, b.len(), r.len()));
                        }
                        PolarsResult::Ok(b)
                    }
                };
                if (range.end as u64) <= span_end && span_end <= MAX_SHARED_PREFIX_BYTES {
                    // The vectors of a file are small and adjacent: one request serves all of
                    // them, and holds the version byte. If it cannot be had (the budget is spent,
                    // or some other vector of the file reaches past its end), only the vectors
                    // that are themselves bad may fail, so this one reads its own frame.
                    let prefix = state
                        .prefix
                        .get_or_init(|| async {
                            let reservation = this.try_reserve(span_end)?;
                            // Copied, so that what is kept for the scan is the counted
                            // `span_end` bytes, whatever backing the store's buffer has.
                            let prefix = get(0..span_end as usize)
                                .await
                                .ok()
                                .map(|b| Buffer::from_vec(b.to_vec()));
                            if prefix.is_some() {
                                reservation.commit();
                            }
                            prefix
                        })
                        .await;
                    if let Some(prefix) = prefix {
                        decoder.check_file_version(&prefix[..1])?;
                        return Ok(prefix.clone().sliced(range));
                    }
                }
                // The prefix read showed the file is too short for this vector: report that, not
                // whatever the store does with a range past the end.
                let known_len = state.known_len.load(std::sync::atomic::Ordering::Relaxed);
                if known_len != u64::MAX && (range.end as u64) > known_len {
                    return Err(short_read_err(
                        &source_path,
                        known_len.saturating_sub(range.start as u64) as usize,
                        range.len(),
                    ));
                }
                // The version byte is validated once per file, before any frame is requested.
                state
                    .version_ok
                    .get_or_try_init(|| async { decoder.check_file_version(&get(0..1).await?) })
                    .await?;
                get(range).await
            }),
        );
        handle.await.unwrap()
    }
}

fn short_read_err(path: &PlRefPath, got: usize, want: usize) -> polars_error::PolarsError {
    polars_err!(
        ComputeError:
        "deletion vector file {path} is shorter than the vector: read {got} of {want} bytes"
    )
}

type NativeDeletionVectorCache =
    PlHashMap<DeltaDeletionVector, Weak<tokio::sync::OnceCell<ExternalFilterMask>>>;

fn native_cache_cell(
    cache: &parking_lot::Mutex<NativeDeletionVectorCache>,
    vector: &DeltaDeletionVector,
) -> Arc<tokio::sync::OnceCell<ExternalFilterMask>> {
    let mut cache = cache.lock();
    let entry = cache.entry(vector.clone()).or_default();
    match entry.upgrade() {
        Some(cell) => cell,
        None => {
            let cell = Arc::new(tokio::sync::OnceCell::new());
            *entry = Arc::downgrade(&cell);
            cell
        },
    }
}

#[cfg(test)]
mod tests {
    use polars_utils::slice_enum::Slice;

    use super::ExternalFilterMask;

    fn split_mask<I>(mask: I, slice: Slice) -> (Vec<bool>, Slice)
    where
        I: IntoIterator<Item = bool>,
    {
        let mask = ExternalFilterMask::Iceberg {
            mask: mask.into_iter().collect(),
        };

        let slice = mask.calc_physical_slice(slice);

        let mask = {
            let range = slice.resolve(mask.get_mask().len());
            mask.get_mask().sliced(range.start, range.len())
        };

        (mask.into_iter().collect(), slice)
    }

    #[test]
    fn native_cache_shares_only_live_loads() {
        use super::*;
        let vector = DeltaDeletionVector {
            location: DeltaDeletionVectorLocation::Inline(Buffer::default()),
            size_in_bytes: 0,
            cardinality: 0,
        };
        let cache = parking_lot::Mutex::new(NativeDeletionVectorCache::default());
        let a = native_cache_cell(&cache, &vector);
        let b = native_cache_cell(&cache, &vector);
        assert!(Arc::ptr_eq(&a, &b));
        let weak = Arc::downgrade(&a);
        drop(a);
        drop(b);
        assert!(weak.upgrade().is_none());
        let c = native_cache_cell(&cache, &vector);
        assert!(!Weak::ptr_eq(&weak, &Arc::downgrade(&c)));
    }

    #[test]
    fn test_split_slice_positive() {
        const T: bool = true;
        const F: bool = false;

        {
            let mask = [];
            let slice = Slice::Positive { offset: 0, len: 0 };
            let (mask, slice) = split_mask(mask, slice);

            assert!(mask.is_empty());
            assert_eq!(slice, Slice::Positive { offset: 0, len: 0 });
        }

        {
            let mask = [];
            let slice = Slice::Positive { offset: 1, len: 1 };
            let (mask, slice) = split_mask(mask, slice);

            assert!(mask.is_empty());
            assert_eq!(slice, Slice::Positive { offset: 1, len: 1 });
        }

        {
            let mask = [F, T];
            let slice = Slice::Positive { offset: 0, len: 0 };
            let (mask, slice) = split_mask(mask, slice);

            assert!(mask.is_empty());
            assert_eq!(slice, Slice::Positive { offset: 1, len: 0 });
        }

        {
            let mask = [F, T];
            let slice = Slice::Positive {
                offset: usize::MAX,
                len: 0,
            };
            let (mask, slice) = split_mask(mask, slice);

            assert!(mask.is_empty());
            assert_eq!(
                slice,
                Slice::Positive {
                    offset: usize::MAX,
                    len: 0
                }
            );
        }

        {
            let mask = [F, T, F, F, T, F, T, F, T];
            //       true_index :     0        1     2     3  (index after deletion)
            //       phys_index :  0, 1, 2, 3, 4, 5, 6, 7, 8
            //           offset :              ^              = 4
            //              len :              ^^^^^^^        = 3 items

            let slice = Slice::Positive { offset: 1, len: 2 };
            let (mask, slice) = split_mask(mask, slice);

            assert_eq!(mask.as_slice(), &[T, F, T]);
            assert_eq!(slice, Slice::Positive { offset: 4, len: 3 });
        }

        {
            let mask = [T, F];
            let slice = Slice::Positive { offset: 0, len: 2 };
            let (mask, slice) = split_mask(mask, slice);

            assert_eq!(mask.as_slice(), &[T, F]);
            assert_eq!(slice, Slice::Positive { offset: 0, len: 3 });
        }

        {
            let mask = [T, F];
            let slice = Slice::Positive {
                offset: 100,
                len: 50,
            };
            let (mask, slice) = split_mask(mask, slice);

            assert!(mask.is_empty());
            assert_eq!(
                slice,
                Slice::Positive {
                    offset: 101,
                    len: 50
                }
            );
        }

        {
            let mask = [T, F];
            let slice = Slice::Positive { offset: 0, len: 50 };
            let (mask, slice) = split_mask(mask, slice);

            assert_eq!(mask.as_slice(), &[T, F]);
            assert_eq!(slice, Slice::Positive { offset: 0, len: 51 });
        }

        {
            let mask = [T, F, F, T, F];
            let slice = Slice::Positive { offset: 0, len: 50 };
            let (mask, slice) = split_mask(mask, slice);

            assert_eq!(mask.as_slice(), &[T, F, F, T, F]);
            assert_eq!(slice, Slice::Positive { offset: 0, len: 53 });
        }
    }
}
