use std::io::{BufReader, Cursor};
use std::sync::{LazyLock, RwLock};

use either::Either;
use polars_buffer::Buffer;
use polars_core::runtime::ASYNC;
use polars_io::cloud::concurrency_config::FetchConfig;
use polars_io::csv::read::streaming::read_until_start_and_infer_schema;
use polars_io::prelude::CsvReadOptions;
#[cfg(feature = "parquet")]
use polars_io::prelude::{FileMetadataRef, ParquetObjectStore, ParquetReader};
use polars_io::utils::byte_source::{ByteSource, DynByteSourceBuilder};
use polars_io::utils::compression::{ByteSourceReader, CompressedReader, SupportedCompression};
#[cfg(feature = "json")]
use polars_io::utils::overwrite_schema;
use polars_io::utils::stream_buf_reader::ReaderSource;
use polars_io::{RowIndex, SerReader};

use super::*;

pub(super) async fn dsl_to_ir(
    sources: ScanSources,
    mut unified_scan_args_box: Box<UnifiedScanArgs>,
    scan_type: Box<FileScanDsl>,
    cached_ir: Arc<Mutex<Option<IR>>>,
    cache_file_info: SourcesToFileInfo,
    verbose: bool,
) -> PolarsResult<()> {
    // Note that the first metadata can still end up being `None` later if the files were
    // filtered from predicate pushdown.
    // Check and drop the lock in its own scope
    let is_not_cached = {
        let cached_ir_guard = cached_ir.lock().unwrap();
        cached_ir_guard.is_none()
    };

    if is_not_cached {
        let unified_scan_args = unified_scan_args_box.as_mut();

        if let Some(hive_schema) = unified_scan_args.hive_options.schema.as_deref() {
            match unified_scan_args.hive_options.enabled {
                // Enable hive_partitioning if it is unspecified but a non-empty hive_schema given
                None if !hive_schema.is_empty() => {
                    unified_scan_args.hive_options.enabled = Some(true)
                },
                // hive_partitioning was explicitly disabled
                Some(false) => polars_bail!(
                    ComputeError:
                    "a hive schema was given but hive_partitioning was disabled"
                ),
                Some(true) | None => {},
            }
        }

        let sources_before_expansion = &sources;

        let sources = match &*scan_type {
            #[cfg(feature = "parquet")]
            FileScanDsl::Parquet { .. } => {
                sources
                    .expand_paths_with_hive_update(unified_scan_args)
                    .await?
            },
            #[cfg(feature = "ipc")]
            FileScanDsl::Ipc { .. } => {
                sources
                    .expand_paths_with_hive_update(unified_scan_args)
                    .await?
            },
            #[cfg(feature = "csv")]
            FileScanDsl::Csv { .. } => sources.expand_paths(unified_scan_args).await?,
            #[cfg(feature = "json")]
            FileScanDsl::NDJson { .. } => sources.expand_paths(unified_scan_args).await?,
            #[cfg(feature = "python")]
            FileScanDsl::PythonDataset { .. } => {
                // There are a lot of places that short-circuit if the paths is empty,
                // so we just give a dummy path here.
                ScanSources::Paths(Buffer::from_iter([PlRefPath::new("PL_PY_DSET")]))
            },
            #[cfg(feature = "scan_lines")]
            FileScanDsl::Lines { .. } => sources.expand_paths(unified_scan_args).await?,
            FileScanDsl::ExpandedPaths { .. } => sources.expand_paths(unified_scan_args).await?,
            FileScanDsl::Anonymous { .. } => sources.clone(),
        };

        // For cloud we must deduplicate files. Serialization/deserialization leads to Arc's losing there
        // sharing.
        let (mut file_info, scan_type_ir) = {
            cache_file_info
                .get_or_insert(
                    &scan_type,
                    &sources,
                    sources_before_expansion,
                    unified_scan_args,
                    verbose,
                )
                .await?
        };

        if unified_scan_args.hive_options.enabled.is_none() {
            // We expect this to be `Some(_)` after this point. If it hasn't been auto-enabled
            // we explicitly set it to disabled.
            unified_scan_args.hive_options.enabled = Some(false);
        }

        let hive_parts = if unified_scan_args.hive_options.enabled.unwrap()
            && let Some(file_schema) = file_info.reader_schema.as_ref()
        {
            let paths = sources
                .as_paths()
                .ok_or_else(|| polars_err!(nyi = "Hive-partitioning of in-memory buffers"))?;

            #[allow(unused_assignments)]
            let mut owned = None;

            hive_partitions_from_paths(
                paths,
                unified_scan_args.hive_options.hive_start_idx,
                unified_scan_args.hive_options.schema.clone(),
                match file_schema {
                    Either::Left(v) => {
                        owned = Some(Schema::from_arrow_schema(v.as_ref()));
                        owned.as_ref().unwrap()
                    },
                    Either::Right(v) => v.as_ref(),
                },
                unified_scan_args.hive_options.try_parse_dates,
            )?
        } else {
            None
        };

        if let Some(ref hive_parts) = hive_parts {
            let hive_schema = hive_parts.schema();
            file_info.update_schema_with_hive_schema(hive_schema.clone());
        } else if let Some(hive_schema) = unified_scan_args.hive_options.schema.clone() {
            // We hit here if we are passed the `hive_schema` to `scan_parquet` but end up with an empty file
            // list during path expansion. In this case we still want to return an empty DataFrame with this
            // schema.
            file_info.update_schema_with_hive_schema(hive_schema);
        }

        if let Some(ref file_path_col) = unified_scan_args.include_file_paths {
            let schema: &mut Schema = Arc::make_mut(&mut file_info.schema);

            if schema.contains(file_path_col) {
                polars_bail!(
                    Duplicate: r#"column name for file paths "{}" conflicts with column name from file"#,
                    file_path_col
                );
            }

            schema.insert_at_index(schema.len(), file_path_col.clone(), DataType::String)?;
        }

        unified_scan_args.projection = if let Some(file_schema) = file_info.reader_schema.as_ref() {
            maybe_init_projection_excluding_hive(
                file_schema,
                hive_parts.as_ref().map(|h| h.schema()),
            )
        } else {
            None
        };

        if let Some(row_index) = &unified_scan_args.row_index {
            let schema = Arc::make_mut(&mut file_info.schema);
            *schema = schema
                .new_inserting_at_index(0, row_index.name.clone(), IDX_DTYPE)
                .unwrap();
        }

        let ir = if sources.is_empty() && !matches!(&(*scan_type), FileScanDsl::Anonymous { .. }) {
            IR::DataFrameScan {
                df: Arc::new(DataFrame::empty_with_schema(&file_info.schema)),
                schema: file_info.schema,
                output_schema: None,
            }
        } else {
            let unified_scan_args = unified_scan_args_box;

            IR::Scan {
                sources,
                file_info,
                hive_parts,
                predicate: None,
                predicate_file_skip_applied: None,
                scan_type: Box::new(scan_type_ir),
                output_schema: None,
                unified_scan_args,
            }
        };

        let mut cached_ir = cached_ir.lock().unwrap();
        cached_ir.replace(ir);
    }

    Ok(())
}

pub(super) fn insert_row_index_to_schema(
    schema: &mut Schema,
    name: PlSmallStr,
) -> PolarsResult<()> {
    if schema.contains(&name) {
        polars_bail!(
            Duplicate:
            "cannot add row_index with name '{}': \
            column already exists in file.",
            name,
        )
    }

    schema.insert_at_index(0, name, IDX_DTYPE)?;

    Ok(())
}

#[cfg(any(feature = "parquet", feature = "ipc"))]
fn prepare_output_schema(
    mut schema: Schema,
    row_index: Option<&RowIndex>,
) -> PolarsResult<SchemaRef> {
    if let Some(rc) = row_index {
        insert_row_index_to_schema(&mut schema, rc.name.clone())?;
    }
    Ok(Arc::new(schema))
}

#[cfg(any(feature = "json", feature = "csv"))]
fn prepare_schemas(
    mut schema: Schema,
    row_index: Option<&RowIndex>,
) -> PolarsResult<(SchemaRef, SchemaRef)> {
    Ok(if let Some(rc) = row_index {
        let reader_schema = schema.clone();
        insert_row_index_to_schema(&mut schema, rc.name.clone())?;
        (Arc::new(reader_schema), Arc::new(schema))
    } else {
        let schema = Arc::new(schema);
        (schema.clone(), schema)
    })
}

#[cfg(feature = "parquet")]
pub(super) async fn parquet_file_info(
    sources: &ScanSources,
    row_index: Option<&RowIndex>,
    #[allow(unused)] cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<(
    FileInfo,
    Option<FileMetadataRef>,
    Option<Arc<[FileMetadataRef]>>,
)> {
    use futures::stream::{FuturesOrdered, FuturesUnordered, StreamExt};
    use polars_core::error::feature_gated;

    let n_sources = sources.len();
    let first_scan_source = sources.iter().next().expect("at least one source");

    // File 0: schema + num_rows + full metadata (schema comes from file 0 only;
    // no cross-file schema evolution). Built as a future so it runs concurrently
    // with the other-source reads below. File 0's schema is not borrowed by the
    // other-source reads, so they are independent and share a single concurrency
    // wave.
    let first_fut = async move {
        if first_scan_source.is_cloud_url() {
            let first_path = first_scan_source.as_path().unwrap();
            feature_gated!("cloud", {
                let mut reader =
                    ParquetObjectStore::from_uri(first_path.clone(), cloud_options, None).await?;
                PolarsResult::Ok((
                    reader.schema().await?,
                    reader.num_rows().await?,
                    reader.get_metadata().await?.clone(),
                ))
            })
        } else {
            let memslice = first_scan_source.to_memslice()?;
            let mut reader = ParquetReader::new(Cursor::new(memslice));
            PolarsResult::Ok((
                reader.schema()?,
                reader.num_rows()?,
                reader.get_metadata()?.clone(),
            ))
        }
    };

    // Resolve metadata for sources past the first, dispatched by
    // `POLARS_RESOLVE_METADATA_LEVEL`:
    // - `None` (default): extrapolate `first_num_rows * n_sources`, no extra reads.
    // - `RowCounts`: per-source thrift field 3 only, exact total.
    // - `Sampled`: read one concurrency wave of footers and extrapolate (exact
    //   when the whole set fits in the wave).
    // - `Full`: per-source footer, populates `metadata_per_source` for the
    //   distributed scheduler.
    //
    // Every multi-source arm reads file 0 concurrently with the other sources
    // via `try_join`, so the schema read does not cost an extra serial round
    // trip; a file-0 error short-circuits and cancels the other reads.
    // Per-scan size resolution. `reader_schema`/`first_metadata` always come
    // from file 0; only the size fields vary by mode.
    struct Resolution {
        reader_schema: ArrowSchemaRef,
        first_metadata: FileMetadataRef,
        metadata_per_source: Option<Arc<[FileMetadataRef]>>,
        known_size: Option<usize>,
        estimated_size: usize,
    }

    let mode = polars_config::config().resolve_metadata_level();
    let resolved = if n_sources == 1 {
        // Only file 0, so its count is exact and there is nothing to join.
        let (reader_schema, first_num_rows, first_metadata) = first_fut.await?;
        Resolution {
            reader_schema,
            first_metadata,
            metadata_per_source: None,
            known_size: Some(first_num_rows),
            estimated_size: first_num_rows,
        }
    } else {
        use polars_config::ResolveMode;
        match mode {
            ResolveMode::None => {
                // No other-source reads; extrapolate from file 0 alone.
                let (reader_schema, first_num_rows, first_metadata) = first_fut.await?;
                Resolution {
                    reader_schema,
                    first_metadata,
                    metadata_per_source: None,
                    known_size: None,
                    estimated_size: first_num_rows.saturating_mul(n_sources),
                }
            },
            ResolveMode::RowCounts => {
                // Every other file's row count (thrift field 3 only), read
                // concurrently with file 0.
                let rest_fut = async move {
                    let mut futures = (1..n_sources)
                        .map(|i| async move {
                            read_parquet_num_rows(sources.at(i), cloud_options).await
                        })
                        .collect::<FuturesUnordered<_>>();

                    // Best-effort: a file that fails to decode at plan time
                    // (e.g. an invalid file in a hive partition not yet pruned)
                    // simply contributes 0. If execution needs the file it
                    // errors then; if predicate pushdown prunes it first, it
                    // never matters.
                    let mut total = 0usize;
                    while let Some(res) = futures.next().await {
                        if let Ok(n) = res {
                            total = total.saturating_add(n as usize);
                        }
                    }
                    PolarsResult::Ok(total)
                };
                let ((reader_schema, first_num_rows, first_metadata), others) =
                    futures::future::try_join(first_fut, rest_fut).await?;
                let total = first_num_rows.saturating_add(others);
                Resolution {
                    reader_schema,
                    first_metadata,
                    metadata_per_source: None,
                    known_size: Some(total),
                    estimated_size: total,
                }
            },
            ResolveMode::Sampled => {
                // Sample `sqrt(n)` footers (`sampled_source_indices`) and
                // extrapolate the per-file mean. Exact when the sample spans
                // every file; otherwise `estimated_size` only, not `known_size`.
                //
                // TODO: byte-weight (`rows_per_byte * total_bytes`, skew-robust)
                // once per-file LIST sizes reach plan time.
                let limit = polars_io::pl_async::get_concurrency_limit() as usize;
                let sample = sampled_source_indices(n_sources, limit);
                // file 0 plus the sample; if that spans every file we read all.
                let read_all = sample.len() + 1 == n_sources;
                let rest_fut = async move {
                    let mut futures = sample
                        .iter()
                        .map(|&i| async move {
                            read_parquet_num_rows(sources.at(i), cloud_options).await
                        })
                        .collect::<FuturesUnordered<_>>();
                    let mut rows = 0usize;
                    let mut read = 0usize;
                    while let Some(res) = futures.next().await {
                        if let Ok(n) = res {
                            rows = rows.saturating_add(n as usize);
                            read += 1;
                        }
                    }
                    PolarsResult::Ok((rows, read))
                };
                let ((reader_schema, first_num_rows, first_metadata), (other_rows, other_read)) =
                    futures::future::try_join(first_fut, rest_fut).await?;
                // file 0 always counts toward the sample.
                let sampled_rows = first_num_rows.saturating_add(other_rows);
                let n_read = 1 + other_read;
                // Extrapolate the per-file mean across all sources. `n_read` is
                // always >= 1 (file 0), so the division is safe; when every file
                // was read successfully this is exact, so report it as known.
                let estimated =
                    ((sampled_rows as u128 * n_sources as u128) / n_read as u128) as usize;
                let known = (read_all && n_read == n_sources).then_some(estimated);
                Resolution {
                    reader_schema,
                    first_metadata,
                    metadata_per_source: None,
                    known_size: known,
                    estimated_size: estimated,
                }
            },
            ResolveMode::Full => {
                // Each file decoded with its own schema: per-file schemas may
                // differ in columns, dtypes, or column order. Read concurrently
                // with file 0; `None` marks a file that failed to decode.
                let rest_fut = async move {
                    let mut futures = (1..n_sources)
                        .map(|i| read_parquet_metadata(sources.at(i), cloud_options))
                        .collect::<FuturesOrdered<_>>();
                    let mut rest: Vec<Option<FileMetadataRef>> = Vec::with_capacity(n_sources - 1);
                    while let Some(file_result) = futures.next().await {
                        rest.push(file_result.ok());
                    }
                    PolarsResult::Ok(rest)
                };
                let ((reader_schema, first_num_rows, first_metadata), rest) =
                    futures::future::try_join(first_fut, rest_fut).await?;

                // Slot 0 satisfies the `metadata_per_source[0] == first_metadata`
                // invariant; a failed read falls back to file 0's metadata (the
                // cloud scheduler re-fetches if it needs accurate row_groups).
                let mut per_file: Vec<FileMetadataRef> = Vec::with_capacity(n_sources);
                per_file.push(first_metadata.clone());
                let mut total: usize = first_num_rows;
                for slot in rest {
                    match slot {
                        Some(m) => {
                            total = total.saturating_add(m.num_rows);
                            per_file.push(m);
                        },
                        None => per_file.push(first_metadata.clone()),
                    }
                }
                let dense: Arc<[FileMetadataRef]> = per_file.into();
                Resolution {
                    reader_schema,
                    first_metadata,
                    metadata_per_source: Some(dense),
                    known_size: Some(total),
                    estimated_size: total,
                }
            },
        }
    };

    let schema = prepare_output_schema(
        Schema::from_arrow_schema(resolved.reader_schema.as_ref()),
        row_index,
    )?;

    let file_info = FileInfo::new(
        schema,
        Some(Either::Left(resolved.reader_schema)),
        (resolved.known_size, resolved.estimated_size),
    );

    Ok((
        file_info,
        Some(resolved.first_metadata),
        resolved.metadata_per_source,
    ))
}

/// Pick an evenly-strided sample of indices in `1..n_sources` for
/// [`ResolveMode::Sampled`] (file 0 is read separately). Size is `sqrt(n)`,
/// floored at `SAMPLE_FLOOR`, capped at `limit` (the shared concurrency budget)
/// so one scan does not starve others, never above the file count.
#[cfg(feature = "parquet")]
fn sampled_source_indices(n_sources: usize, limit: usize) -> Vec<usize> {
    // Minimum sample so a small scan still extrapolates from enough files.
    const SAMPLE_FLOOR: usize = 16;
    // `k` = total footers this wave (incl. file 0); `limit` last so it stays a
    // hard ceiling.
    let k = ((n_sources as f64).sqrt().ceil() as usize)
        .max(SAMPLE_FLOOR)
        .min(limit.max(1))
        .min(n_sources);
    if k <= 1 {
        return Vec::new();
    }
    // `k - 1` more, evenly strided over `1..n_sources`.
    let extra = k - 1;
    let span = n_sources - 1;
    (0..extra).map(|j| 1 + (j * span) / extra).collect()
}

/// Fetch one source's full footer. Used by [`parquet_file_info`] in
/// `Full` resolve mode.
#[cfg(feature = "parquet")]
async fn read_parquet_metadata(
    source: ScanSourceRef<'_>,
    #[allow(unused)] cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<FileMetadataRef> {
    use polars_core::error::feature_gated;

    if source.is_cloud_url() {
        let path = source.as_path().unwrap();
        feature_gated!("cloud", {
            let mut reader =
                ParquetObjectStore::from_uri(path.clone(), cloud_options, None).await?;
            reader.get_metadata().await.cloned()
        })
    } else {
        let memslice = source.to_memslice()?;
        let mut cursor = Cursor::new(memslice);
        let md = polars_parquet::parquet::read::read_metadata(&mut cursor)?;
        Ok(Arc::new(md))
    }
}

/// Fetch one source's `num_rows` (thrift field 3 only); skips
/// schema, row_groups, and the rest. Used by [`parquet_file_info`]
/// in `RowCounts` resolve mode.
#[cfg(feature = "parquet")]
async fn read_parquet_num_rows(
    source: ScanSourceRef<'_>,
    #[allow(unused)] cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<i64> {
    use polars_core::error::feature_gated;

    if source.is_cloud_url() {
        let path = source.as_path().unwrap();
        feature_gated!("cloud", {
            let mut reader =
                ParquetObjectStore::from_uri(path.clone(), cloud_options, None).await?;
            reader.num_rows_only().await
        })
    } else {
        let memslice = source.to_memslice()?;
        let mut cursor = Cursor::new(memslice);
        polars_parquet::parquet::read::read_num_rows(&mut cursor).map_err(Into::into)
    }
}

pub fn max_metadata_scan_cached() -> usize {
    static MAX_SCANS_METADATA_CACHED: LazyLock<usize> = LazyLock::new(|| {
        let value = std::env::var("POLARS_MAX_CACHED_METADATA_SCANS").map_or(8, |v| {
            v.parse::<usize>()
                .expect("invalid `POLARS_MAX_CACHED_METADATA_SCANS` value")
        });
        if value == 0 {
            return usize::MAX;
        }
        value
    });
    *MAX_SCANS_METADATA_CACHED
}

// TODO! return metadata arced
#[cfg(feature = "ipc")]
pub(super) async fn ipc_file_info(
    first_scan_source: ScanSourceRef<'_>,
    row_index: Option<&RowIndex>,
    cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<(FileInfo, arrow::io::ipc::read::FileMetadata)> {
    use polars_core::error::feature_gated;

    let metadata = match first_scan_source {
        ScanSourceRef::Path(path) => {
            if path.has_scheme() {
                feature_gated!("cloud", {
                    polars_io::ipc::IpcReaderAsync::from_uri(path.clone(), cloud_options)
                        .await?
                        .metadata()
                        .await?
                })
            } else {
                arrow::io::ipc::read::read_file_metadata(&mut std::io::BufReader::new(
                    polars_utils::open_file(path.as_std_path())?,
                ))?
            }
        },
        ScanSourceRef::File(file) => {
            arrow::io::ipc::read::read_file_metadata(&mut std::io::BufReader::new(file))?
        },
        ScanSourceRef::Buffer(buff) => {
            arrow::io::ipc::read::read_file_metadata(&mut std::io::Cursor::new(buff))?
        },
    };

    let file_info = FileInfo::new(
        prepare_output_schema(
            Schema::from_arrow_schema(metadata.schema.as_ref()),
            row_index,
        )?,
        Some(Either::Left(Arc::clone(&metadata.schema))),
        (None, usize::MAX),
    );

    Ok((file_info, metadata))
}

#[cfg(feature = "csv")]
pub async fn csv_file_info(
    sources: &ScanSources,
    _first_scan_source: ScanSourceRef<'_>,
    row_index: Option<&RowIndex>,
    csv_options: &mut CsvReadOptions,
    cloud_options: Option<&polars_io::cloud::CloudOptions>,
    missing_columns_policy: MissingColumnsPolicy,
) -> PolarsResult<FileInfo> {
    use polars_core::error::feature_gated;
    use polars_core::runtime::RAYON;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    // Holding _first_scan_source should guarantee sources is not empty.
    debug_assert!(!sources.is_empty());

    // TODO:
    // * See if we can do better than scanning all files if there is a row limit

    // prints the error message if paths is empty.
    let run_async =
        sources.is_cloud_url() || (sources.is_paths() && polars_config::config().force_async());

    let cache_entries = {
        if run_async {
            let sources = sources.clone();
            assert!(sources.as_paths().is_some());

            feature_gated!("cloud", {
                Some(
                    polars_io::file_cache::init_entries_from_uri_list(
                        (0..sources.len())
                            .map(move |i| sources.as_paths().unwrap().get(i).unwrap().clone()),
                        cloud_options,
                    )
                    .await?,
                )
            })
        } else {
            None
        }
    };

    let infer_schema_length = csv_options.infer_schema_length;
    let infer_schema_func = |i| {
        const ASSUMED_COMPRESSION_RATIO: usize = 4;
        let source = sources.at(i);

        let (mem_slice_raw, file_size, decompressed_slice_size_hint) = if run_async
            && let Some(infer_schema_length) = infer_schema_length
        {
            // Only download what we need for schema inference.
            // To do so, we use an iterative two-way progressive trial-and-error download strategy
            // until we either have enough rows, or reached EOF. In every iteration, we either
            // increase fetch_size (download progressively more), or try_read_size (try and
            // decompress more of what we have, in the case of compressed).
            const INITIAL_FETCH: usize = 64 * 1024;

            // Collect metadata.
            let byte_source = ASYNC.block_on(async move {
                source
                    .to_dyn_byte_source(
                        &DynByteSourceBuilder::ObjectStore(FetchConfig::streaming()),
                        cloud_options,
                        None,
                    )
                    .await
            })?;
            let byte_source = Arc::new(byte_source);

            let file_size = {
                let byte_source = byte_source.clone();
                ASYNC.block_on(async move { byte_source.get_size().await })?
            };

            let compression = if file_size >= 4 {
                let byte_source = byte_source.clone();
                let magic_range = 0..4;
                let magic_bytes =
                    ASYNC.block_on(async move { byte_source.get_range(magic_range).await })?;
                SupportedCompression::check(&magic_bytes)
            } else {
                None
            };

            let mut offset = 0;
            let mut fetch_size = INITIAL_FETCH;
            let mut try_read_size = INITIAL_FETCH * ASSUMED_COMPRESSION_RATIO;
            let mut truncated_bytes: Vec<u8> = Vec::with_capacity(INITIAL_FETCH);
            let mut reached_eof = false;

            // Collect enough rows to satisfy infer_schema_length.
            let (mem_slice_raw, decompressed_slice_size_hint) = loop {
                let range = offset..std::cmp::min(file_size, offset + fetch_size);

                if range.is_empty() {
                    reached_eof = true
                } else {
                    let byte_source = byte_source.clone();
                    let fetch_bytes =
                        ASYNC.block_on(async move { byte_source.get_range(range).await })?;
                    offset += fetch_bytes.len();
                    truncated_bytes.extend_from_slice(fetch_bytes.as_ref());
                }

                let decompressed_size_hint =
                    Some(offset * compression.map_or(1, |_| ASSUMED_COMPRESSION_RATIO));
                let mut reader = ByteSourceReader::<ReaderSource>::from_memory(
                    Buffer::from_owner(truncated_bytes.clone()),
                )?;

                let read_size = if compression.is_none() {
                    offset
                } else if reached_eof {
                    usize::MAX
                } else {
                    try_read_size
                };

                // Note: if `count_rows_from_reader_par` and therefore also `read_next_slice` were to
                // handle truncated compressed bytes gracefully, we could avoid the following EoF check
                // and remove `try_read_size` from the loop.
                let (slice, bytes_read) =
                    match reader.read_next_slice(&Buffer::new(), read_size, decompressed_size_hint)
                    {
                        Ok(v) => v,
                        // We assume that unexpected EOF indicates that we lack sufficient data.
                        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                            fetch_size *= 2;
                            continue;
                        },
                        Err(e) => Err(e)?,
                    };

                let row_count = polars_io::csv::read::count_rows_from_slice_par(
                    slice.clone(),
                    csv_options.parse_options.quote_char,
                    csv_options.parse_options.comment_prefix.as_ref(),
                    csv_options.parse_options.eol_char,
                    csv_options.has_header,
                    csv_options.skip_lines,
                    csv_options.skip_rows,
                    csv_options.skip_rows_after_header,
                    csv_options.raise_if_empty,
                )?;

                if row_count < infer_schema_length && !reached_eof {
                    if compression.is_some() && bytes_read == read_size {
                        // Decompressor had more to give — read_size too small
                        try_read_size *= 2;
                    } else {
                        // Decompressor exhausted input — need more compressed bytes
                        // Or, no compression
                        fetch_size *= 2;
                    }
                    continue;
                }

                break (Buffer::from_owner(truncated_bytes), Some(bytes_read));
            };
            (mem_slice_raw, file_size, decompressed_slice_size_hint)
        } else {
            let mem_slice_raw =
                source.to_buffer_possibly_async(run_async, cache_entries.as_ref(), i)?;
            let file_size = mem_slice_raw.len();
            let compression = SupportedCompression::check(&mem_slice_raw);
            let decompressed_slice_size_hint = Some(match compression {
                None => file_size,
                Some(_) => file_size * ASSUMED_COMPRESSION_RATIO,
            });
            (mem_slice_raw, file_size, decompressed_slice_size_hint)
        };

        let mut reader = ByteSourceReader::from_memory(mem_slice_raw)?;
        let compression = reader.compression();

        let mut first_row_len = 0;
        let (schema, _) = read_until_start_and_infer_schema(
            csv_options,
            None,
            decompressed_slice_size_hint,
            Some(Box::new(|line| {
                first_row_len = line.len() + 1;
            })),
            &mut reader,
        )?;

        let decompressed_file_size_hint = match compression {
            None => file_size,
            Some(_) => file_size * ASSUMED_COMPRESSION_RATIO,
        };

        // TODO. We can do (much) better by collect statistics as part of row count and/or schema
        // inference, including observed average row_length and compression ratio.
        let estimated_rows =
            (decompressed_file_size_hint as f64 / first_row_len as f64).round() as usize;

        Ok((schema, estimated_rows))
    };

    let merge_func =
        |a: PolarsResult<(Schema, usize)>, b: PolarsResult<(Schema, usize)>| match (a, b) {
            (Err(e), _) | (_, Err(e)) => Err(e),
            (Ok((mut schema_a, row_estimate_a)), Ok((schema_b, row_estimate_b))) => {
                match (schema_a.is_empty(), schema_b.is_empty()) {
                    (true, _) => Ok((schema_b, row_estimate_b)),
                    (_, true) => Ok((schema_a, row_estimate_a)),
                    _ => match missing_columns_policy {
                        MissingColumnsPolicy::Raise => {
                            schema_a.to_supertype(&schema_b)?;
                            Ok((schema_a, row_estimate_a.saturating_add(row_estimate_b)))
                        },
                        MissingColumnsPolicy::Insert => {
                            // Union merge: keep all columns from both schemas,
                            // supertype columns that exist in both.
                            use polars_core::utils::try_get_supertype;
                            for (name, dtype) in schema_b.iter() {
                                match schema_a.get(name) {
                                    Some(existing_dtype) => {
                                        let st = try_get_supertype(existing_dtype, dtype)?;
                                        schema_a.with_column(name.clone(), st);
                                    },
                                    None => {
                                        schema_a.with_column(name.clone(), dtype.clone());
                                    },
                                }
                            }
                            Ok((schema_a, row_estimate_a.saturating_add(row_estimate_b)))
                        },
                    },
                }
            },
        };

    assert!(
        csv_options.schema.is_none(),
        "DSL to IR schema inference should not run if user provides a schema."
    );
    // Run inference in parallel with a specific merge order.
    // TODO: flatten to single level once Schema::to_supertype is commutative.
    let si_results = RAYON.join(
        || infer_schema_func(0),
        || {
            (1..sources.len())
                .into_par_iter()
                .map(infer_schema_func)
                .reduce(|| Ok(Default::default()), merge_func)
        },
    );

    let (inferred_schema, estimated_n_rows) = merge_func(si_results.0, si_results.1)?;
    let inferred_schema_ref = Arc::new(inferred_schema);

    let (schema, reader_schema) = if let Some(rc) = row_index {
        let mut output_schema = (*inferred_schema_ref).clone();
        insert_row_index_to_schema(&mut output_schema, rc.name.clone())?;

        (Arc::new(output_schema), inferred_schema_ref)
    } else {
        (inferred_schema_ref.clone(), inferred_schema_ref)
    };

    Ok(FileInfo::new(
        schema,
        Some(Either::Right(reader_schema)),
        (None, estimated_n_rows),
    ))
}

#[cfg(feature = "json")]
pub async fn ndjson_file_info(
    sources: &ScanSources,
    first_scan_source: ScanSourceRef<'_>,
    row_index: Option<&RowIndex>,
    ndjson_options: &NDJsonReadOptions,
    cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<FileInfo> {
    use polars_core::error::feature_gated;

    let run_async =
        sources.is_cloud_url() || (sources.is_paths() && polars_config::config().force_async());

    let cache_entries = {
        if run_async {
            let sources = sources.clone();
            assert!(sources.as_paths().is_some());

            feature_gated!("cloud", {
                Some(
                    polars_io::file_cache::init_entries_from_uri_list(
                        (0..sources.len())
                            .map(move |i| sources.as_paths().unwrap().get(i).unwrap().clone()),
                        cloud_options,
                    )
                    .await?,
                )
            })
        } else {
            None
        }
    };

    let infer_schema_length = ndjson_options.infer_schema_length;

    let mut schema = if let Some(schema) = ndjson_options.schema.clone() {
        schema
    } else if run_async && let Some(infer_schema_length) = infer_schema_length {
        // Only download what we need for schema inference.
        // To do so, we use an iterative two-way progressive trial-and-error download strategy
        // until we either have enough rows, or reached EOF. In every iteration, we either
        // increase fetch_size (download progressively more), or try_read_size (try and
        // decompress more of what we have, in the case of compressed).
        use polars_io::utils::compression::{ByteSourceReader, SupportedCompression};
        use polars_io::utils::stream_buf_reader::ReaderSource;

        const INITIAL_FETCH: usize = 64 * 1024;
        const ASSUMED_COMPRESSION_RATIO: usize = 4;

        let first_scan_source = first_scan_source.into_owned()?.clone();
        let cloud_options = cloud_options.cloned();
        // TODO. Support IOMetrics collection during planning phase.
        let byte_source = ASYNC
            .spawn(async move {
                first_scan_source
                    .as_scan_source_ref()
                    .to_dyn_byte_source(
                        &DynByteSourceBuilder::ObjectStore(FetchConfig::streaming()),
                        cloud_options.as_ref(),
                        None,
                    )
                    .await
            })
            .await
            .unwrap()?;
        let byte_source = Arc::new(byte_source);

        let file_size = {
            let byte_source = byte_source.clone();
            ASYNC
                .spawn(async move { byte_source.get_size().await })
                .await
                .unwrap()?
        };

        let mut offset = 0;
        let mut fetch_size = INITIAL_FETCH;
        let mut try_read_size = INITIAL_FETCH * ASSUMED_COMPRESSION_RATIO;
        let mut truncated_bytes: Vec<u8> = Vec::with_capacity(INITIAL_FETCH);
        let mut reached_eof = false;

        // Collect enough rows to satisfy infer_schema_length
        let memslice = loop {
            let range = offset..std::cmp::min(file_size, offset + fetch_size);

            if range.is_empty() {
                reached_eof = true
            } else {
                let byte_source = byte_source.clone();
                let fetch_bytes = ASYNC
                    .spawn(async move { byte_source.get_range(range).await })
                    .await
                    .unwrap()?;
                offset += fetch_bytes.len();
                truncated_bytes.extend_from_slice(fetch_bytes.as_ref());
            }

            let compression = SupportedCompression::check(&truncated_bytes);
            let mut reader = ByteSourceReader::<ReaderSource>::from_memory(Buffer::from_owner(
                truncated_bytes.clone(),
            ))?;
            let read_size = if compression.is_none() {
                offset
            } else if reached_eof {
                usize::MAX
            } else {
                try_read_size
            };

            let uncompressed_size_hint = Some(
                offset
                    * if compression.is_none() {
                        1
                    } else {
                        ASSUMED_COMPRESSION_RATIO
                    },
            );

            let (slice, bytes_read) =
                match reader.read_next_slice(&Buffer::new(), read_size, uncompressed_size_hint) {
                    Ok(v) => v,
                    // We assume that unexpected EOF indicates that we lack sufficient data.
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        fetch_size *= 2;
                        continue;
                    },
                    Err(e) => Err(e)?,
                };

            if polars_io::ndjson::count_rows(&slice) < infer_schema_length.into() && !reached_eof {
                if compression.is_some() && bytes_read == read_size {
                    // Decompressor had more to give — read_size too small
                    try_read_size *= 2;
                } else {
                    // Decompressor exhausted input — need more compressed bytes
                    // Or, no compression
                    fetch_size *= 2;
                }
                continue;
            }

            break slice;
        };

        let mut buf_reader = BufReader::new(Cursor::new(memslice));
        Arc::new(polars_io::ndjson::infer_schema(
            &mut buf_reader,
            ndjson_options.infer_schema_length,
        )?)
    } else {
        // Download the entire object.
        // Warning - this is potentially memory-expensive in the case of a cloud source, and goes
        // against the design goal of a streaming reader. This can be optimized.
        let mem_slice =
            first_scan_source.to_buffer_possibly_async(run_async, cache_entries.as_ref(), 0)?;
        let mut reader = BufReader::new(CompressedReader::try_new(mem_slice)?);

        Arc::new(polars_io::ndjson::infer_schema(
            &mut reader,
            ndjson_options.infer_schema_length,
        )?)
    };

    if let Some(overwriting_schema) = &ndjson_options.schema_overwrite {
        overwrite_schema(Arc::make_mut(&mut schema), overwriting_schema)?;
    }

    let mut reader_schema = schema.clone();

    if row_index.is_some() {
        (schema, reader_schema) = prepare_schemas(Arc::unwrap_or_clone(schema), row_index)?
    }

    Ok(FileInfo::new(
        schema,
        Some(Either::Right(reader_schema)),
        (None, usize::MAX),
    ))
}

// Add flags that influence metadata/schema here
#[derive(Eq, Hash, PartialEq)]
enum CachedSourceKey {
    ParquetIpc {
        first_path: PlRefPath,
        schema_overwrite: Option<SchemaRef>,
    },
    CsvJson {
        paths: Buffer<PlRefPath>,
        schema: Option<SchemaRef>,
        schema_overwrite: Option<SchemaRef>,
    },
}

#[derive(Default, Clone)]
pub(super) struct SourcesToFileInfo {
    inner: Arc<RwLock<PlHashMap<CachedSourceKey, (FileInfo, FileScanIR)>>>,
}

impl SourcesToFileInfo {
    async fn infer_or_parse(
        &self,
        scan_type: FileScanDsl,
        sources: &ScanSources,
        sources_before_expansion: &ScanSources,
        unified_scan_args: &mut UnifiedScanArgs,
    ) -> PolarsResult<(FileInfo, FileScanIR)> {
        let require_first_source = |failed_operation_name: &'static str, hint: &'static str| {
            sources.first_or_empty_expand_err(
                failed_operation_name,
                sources_before_expansion,
                unified_scan_args.glob,
                hint,
            )
        };

        let exact_row_estimation = unified_scan_args.row_count.map(|(total, deleted)| {
            let n: usize = (total - deleted) as usize;
            ((Some(n)), n)
        });
        const DEFAULT_ROW_ESTIMATION: (Option<usize>, usize) = (None, usize::MAX);

        let cloud_options = unified_scan_args.cloud_options.as_ref();

        Ok(match scan_type {
            #[cfg(feature = "parquet")]
            FileScanDsl::Parquet { options } => {
                if let Some(schema) = &options.schema {
                    // We were passed a schema, we don't have to call `parquet_file_info`,
                    // but this does mean we don't have `row_estimation` and `first_metadata`.

                    (
                        FileInfo {
                            schema: schema.clone(),
                            reader_schema: Some(either::Either::Left(Arc::new(
                                schema.to_arrow(CompatLevel::newest()),
                            ))),
                            row_estimation: exact_row_estimation.unwrap_or(DEFAULT_ROW_ESTIMATION),
                        },
                        FileScanIR::Parquet {
                            options,
                            // Schema was passed in; no footer was resolved.
                            first_metadata: None,
                            metadata_per_source: None,
                        },
                    )
                } else {
                    {
                        let first_scan_source = require_first_source(
                            "failed to retrieve first file schema (parquet)",
                            "\
passing a schema can allow \
this scan to succeed with an empty DataFrame.",
                        )?;

                        if verbose() {
                            eprintln!(
                                "sourcing parquet scan file schema from: '{}'",
                                first_scan_source.to_include_path_name()
                            )
                        }

                        let (mut file_info, mut first_metadata, mut metadata_per_source) =
                            scans::parquet_file_info(
                                sources,
                                unified_scan_args.row_index.as_ref(),
                                cloud_options,
                            )
                            .await?;

                        if let Some(exact_row_estimation) = exact_row_estimation {
                            file_info.row_estimation = exact_row_estimation;
                        }

                        if self.inner.read().unwrap().len() > max_metadata_scan_cached() {
                            // Cache pressure: drop both pre-decoded slots so
                            // we don't blow memory. Streaming readers fall
                            // back to fetching footers at scan time.
                            first_metadata = None;
                            metadata_per_source = None;
                        }

                        PolarsResult::Ok((
                            file_info,
                            FileScanIR::Parquet {
                                options,
                                first_metadata,
                                metadata_per_source,
                            },
                        ))
                    }
                    .map_err(|e| e.context(failed_here!(parquet scan)))?
                }
            },
            #[cfg(feature = "ipc")]
            FileScanDsl::Ipc { options } => {
                let first_scan_source =
                    require_first_source("failed to retrieve first file schema (ipc)", "")?;

                if verbose() {
                    eprintln!(
                        "sourcing ipc scan file schema from: '{}'",
                        first_scan_source.to_include_path_name()
                    )
                }

                let (mut file_info, md) = scans::ipc_file_info(
                    first_scan_source,
                    unified_scan_args.row_index.as_ref(),
                    cloud_options,
                )
                .await?;

                if let Some(exact_row_estimation) = exact_row_estimation {
                    file_info.row_estimation = exact_row_estimation;
                }

                PolarsResult::Ok((
                    file_info,
                    FileScanIR::Ipc {
                        options,
                        metadata: Some(Arc::new(md)),
                    },
                ))
            }
            .map_err(|e| e.context(failed_here!(ipc scan)))?,
            #[cfg(feature = "csv")]
            FileScanDsl::Csv { mut options } => {
                let mut file_info = if let Some(schema) = options.schema.clone() {
                    FileInfo {
                        schema: schema.clone(),
                        reader_schema: Some(either::Either::Right(schema)),
                        row_estimation: exact_row_estimation.unwrap_or(DEFAULT_ROW_ESTIMATION),
                    }
                } else {
                    let first_scan_source =
                        require_first_source("failed to retrieve file schemas (csv)", "")?;

                    if verbose() {
                        eprintln!(
                            "sourcing csv scan file schema from: '{}'",
                            first_scan_source.to_include_path_name()
                        )
                    }

                    scans::csv_file_info(
                        sources,
                        first_scan_source,
                        unified_scan_args.row_index.as_ref(),
                        Arc::make_mut(&mut options),
                        cloud_options,
                        unified_scan_args.missing_columns_policy,
                    )
                    .await?
                };

                if let Some(exact_row_estimation) = exact_row_estimation {
                    file_info.row_estimation = exact_row_estimation;
                }

                PolarsResult::Ok((file_info, FileScanIR::Csv { options }))
            }
            .map_err(|e| e.context(failed_here!(csv scan)))?,
            #[cfg(feature = "json")]
            FileScanDsl::NDJson { options } => {
                let mut file_info = if let Some(schema) = options.schema.clone() {
                    FileInfo {
                        schema: schema.clone(),
                        reader_schema: Some(either::Either::Right(schema)),
                        row_estimation: exact_row_estimation.unwrap_or(DEFAULT_ROW_ESTIMATION),
                    }
                } else {
                    let first_scan_source =
                        require_first_source("failed to retrieve first file schema (ndjson)", "")?;

                    if verbose() {
                        eprintln!(
                            "sourcing ndjson scan file schema from: '{}'",
                            first_scan_source.to_include_path_name()
                        )
                    }

                    scans::ndjson_file_info(
                        sources,
                        first_scan_source,
                        unified_scan_args.row_index.as_ref(),
                        &options,
                        cloud_options,
                    )
                    .await?
                };

                if let Some(exact_row_estimation) = exact_row_estimation {
                    file_info.row_estimation = exact_row_estimation;
                }

                PolarsResult::Ok((file_info, FileScanIR::NDJson { options }))
            }
            .map_err(|e| e.context(failed_here!(ndjson scan)))?,
            #[cfg(feature = "python")]
            FileScanDsl::PythonDataset { dataset_object } => (|| {
                if crate::dsl::DATASET_PROVIDER_VTABLE.get().is_none() {
                    polars_bail!(ComputeError: "DATASET_PROVIDER_VTABLE (python) not initialized")
                }

                let mut schema = dataset_object.schema()?;
                let reader_schema = schema.clone();

                if let Some(row_index) = &unified_scan_args.row_index {
                    insert_row_index_to_schema(Arc::make_mut(&mut schema), row_index.name.clone())?;
                }

                PolarsResult::Ok((
                    FileInfo {
                        schema,
                        reader_schema: Some(either::Either::Right(reader_schema)),
                        row_estimation: exact_row_estimation.unwrap_or(DEFAULT_ROW_ESTIMATION),
                    },
                    FileScanIR::PythonDataset {
                        dataset_object,
                        cached_ir: Default::default(),
                    },
                ))
            })()
            .map_err(|e| e.context(failed_here!(python dataset scan)))?,
            #[cfg(feature = "scan_lines")]
            FileScanDsl::Lines { name } => {
                let schema = Arc::new(Schema::from_iter([(name.clone(), DataType::String)]));

                (
                    FileInfo {
                        schema: schema.clone(),
                        reader_schema: Some(either::Either::Right(schema.clone())),
                        row_estimation: exact_row_estimation.unwrap_or(DEFAULT_ROW_ESTIMATION),
                    },
                    FileScanIR::Lines { name },
                )
            },
            FileScanDsl::ExpandedPaths { name } => {
                let schema = Arc::new(Schema::from_iter([(name.clone(), DataType::String)]));

                (
                    FileInfo {
                        schema: schema.clone(),
                        reader_schema: Some(either::Either::Right(schema.clone())),
                        row_estimation: (Some(sources.len()), sources.len()),
                    },
                    FileScanIR::ExpandedPaths { name },
                )
            },
            FileScanDsl::Anonymous {
                mut file_info,
                options,
                function,
            } => {
                if let Some(exact_row_estimation) = exact_row_estimation {
                    file_info.row_estimation = exact_row_estimation;
                }

                (file_info, FileScanIR::Anonymous { options, function })
            },
        })
    }

    pub(super) async fn get_or_insert(
        &self,
        scan_type: &FileScanDsl,
        sources: &ScanSources,
        sources_before_expansion: &ScanSources,
        unified_scan_args: &mut UnifiedScanArgs,
        verbose: bool,
    ) -> PolarsResult<(FileInfo, FileScanIR)> {
        // Only cache non-empty paths. Others are directly parsed.
        let paths = match sources {
            ScanSources::Paths(paths) if !paths.is_empty() => paths.clone(),

            _ => {
                return self
                    .infer_or_parse(
                        scan_type.clone(),
                        sources,
                        sources_before_expansion,
                        unified_scan_args,
                    )
                    .await;
            },
        };

        let (k, v): (CachedSourceKey, Option<(FileInfo, FileScanIR)>) = match scan_type {
            #[cfg(feature = "parquet")]
            FileScanDsl::Parquet { options } => {
                let key = CachedSourceKey::ParquetIpc {
                    first_path: paths[0].clone(),
                    schema_overwrite: options.schema.clone(),
                };

                let guard = self.inner.read().unwrap();
                let v = guard.get(&key);
                (key, v.cloned())
            },
            #[cfg(feature = "ipc")]
            FileScanDsl::Ipc { options: _ } => {
                let key = CachedSourceKey::ParquetIpc {
                    first_path: paths[0].clone(),
                    schema_overwrite: None,
                };

                let guard = self.inner.read().unwrap();
                let v = guard.get(&key);
                (key, v.cloned())
            },
            #[cfg(feature = "csv")]
            FileScanDsl::Csv { options } => {
                let key = CachedSourceKey::CsvJson {
                    paths: paths.clone(),
                    schema: options.schema.clone(),
                    schema_overwrite: options.schema_overwrite.clone(),
                };
                let guard = self.inner.read().unwrap();
                let v = guard.get(&key);
                (key, v.cloned())
            },
            #[cfg(feature = "json")]
            FileScanDsl::NDJson { options } => {
                let key = CachedSourceKey::CsvJson {
                    paths: paths.clone(),
                    schema: options.schema.clone(),
                    schema_overwrite: options.schema_overwrite.clone(),
                };
                let guard = self.inner.read().unwrap();
                let v = guard.get(&key);
                (key, v.cloned())
            },
            _ => {
                return self
                    .infer_or_parse(
                        scan_type.clone(),
                        sources,
                        sources_before_expansion,
                        unified_scan_args,
                    )
                    .await;
            },
        };

        if let Some(out) = v {
            if verbose {
                eprintln!("FILE_INFO CACHE HIT")
            }
            Ok(out)
        } else {
            let v = self
                .infer_or_parse(
                    scan_type.clone(),
                    sources,
                    sources_before_expansion,
                    unified_scan_args,
                )
                .await?;
            self.inner.write().unwrap().insert(k, v.clone());
            Ok(v)
        }
    }
}
