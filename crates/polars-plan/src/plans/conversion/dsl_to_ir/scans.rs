use either::Either;
use polars_io::RowIndex;
use polars_io::cloud::CloudOptions;
#[cfg(feature = "cloud")]
use polars_io::pl_async::get_runtime;
use polars_io::prelude::*;
use polars_io::utils::compression::maybe_decompress_bytes;

use super::*;

pub(super) fn dsl_to_ir(
    sources: ScanSources,
    mut unified_scan_args_box: Box<UnifiedScanArgs>,
    scan_type: Box<FileScanDsl>,
    cached_ir: Arc<Mutex<Option<IR>>>,
    ctxt: &mut DslConversionContext,
) -> PolarsResult<IR> {
    // Note that the first metadata can still end up being `None` later if the files were
    // filtered from predicate pushdown.
    let mut cached_ir = cached_ir.lock().unwrap();

    if cached_ir.is_none() {
        let cloud_options = unified_scan_args_box.cloud_options.clone();
        let cloud_options = cloud_options.as_ref();

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

        let sources = match &*scan_type {
            #[cfg(feature = "parquet")]
            FileScanDsl::Parquet { .. } => {
                sources.expand_paths_with_hive_update(unified_scan_args, cloud_options)?
            },
            #[cfg(feature = "ipc")]
            FileScanDsl::Ipc { .. } => {
                sources.expand_paths_with_hive_update(unified_scan_args, cloud_options)?
            },
            #[cfg(feature = "csv")]
            FileScanDsl::Csv { .. } => sources.expand_paths(unified_scan_args, cloud_options)?,
            #[cfg(feature = "json")]
            FileScanDsl::NDJson { .. } => sources.expand_paths(unified_scan_args, cloud_options)?,
            #[cfg(feature = "python")]
            FileScanDsl::PythonDataset { .. } => {
                // There are a lot of places that short-circuit if the paths is empty,
                // so we just give a dummy path here.
                ScanSources::Paths(Arc::from([PlPath::from_str("dummy")]))
            },
            FileScanDsl::Anonymous { .. } => sources,
        };

        // For cloud we must deduplicate files. Serialization/deserialization leads to Arc's losing there
        // sharing.
        let (mut file_info, scan_type_ir) = ctxt.cache_file_info.get_or_insert(
            &scan_type,
            &sources,
            unified_scan_args,
            cloud_options,
            ctxt.verbose,
        )?;

        if unified_scan_args.hive_options.enabled.is_none() {
            // We expect this to be `Some(_)` after this point. If it hasn't been auto-enabled
            // we explicitly set it to disabled.
            unified_scan_args.hive_options.enabled = Some(false);
        }

        let hive_parts = if unified_scan_args.hive_options.enabled.unwrap()
            && file_info.reader_schema.is_some()
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
                match file_info.reader_schema.as_ref().unwrap() {
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

        unified_scan_args.projection = if file_info.reader_schema.is_some() {
            maybe_init_projection_excluding_hive(
                file_info.reader_schema.as_ref().unwrap(),
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
                scan_type: Box::new(scan_type_ir),
                output_schema: None,
                unified_scan_args,
            }
        };

        cached_ir.replace(ir);
    }

    Ok(cached_ir.clone().unwrap())
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
pub(super) fn parquet_file_info(
    sources: &ScanSources,
    row_index: Option<&RowIndex>,
    #[allow(unused)] cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<(FileInfo, Option<FileMetadataRef>)> {
    use polars_core::error::feature_gated;

    let (reader_schema, num_rows, metadata) = {
        if sources.is_cloud_url() {
            let first_path = &sources.first_path().unwrap();
            feature_gated!("cloud", {
                let uri = first_path.to_str();
                get_runtime().block_in_place_on(async {
                    let mut reader = ParquetObjectStore::from_uri(uri, cloud_options, None).await?;

                    PolarsResult::Ok((
                        reader.schema().await?,
                        Some(reader.num_rows().await?),
                        Some(reader.get_metadata().await?.clone()),
                    ))
                })?
            })
        } else {
            let first_source = sources
                .first()
                .ok_or_else(|| polars_err!(ComputeError: "expected at least 1 source"))?;
            let memslice = first_source.to_memslice()?;
            let mut reader = ParquetReader::new(std::io::Cursor::new(memslice));
            (
                reader.schema()?,
                Some(reader.num_rows()?),
                Some(reader.get_metadata()?.clone()),
            )
        }
    };

    let schema =
        prepare_output_schema(Schema::from_arrow_schema(reader_schema.as_ref()), row_index)?;

    let file_info = FileInfo::new(
        schema,
        Some(Either::Left(reader_schema)),
        (num_rows, num_rows.unwrap_or(0)),
    );

    Ok((file_info, metadata))
}

// TODO! return metadata arced
#[cfg(feature = "ipc")]
pub(super) fn ipc_file_info(
    sources: &ScanSources,
    row_index: Option<&RowIndex>,
    cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<(FileInfo, arrow::io::ipc::read::FileMetadata)> {
    use polars_core::error::feature_gated;
    use polars_utils::plpath::PlPathRef;

    let Some(first) = sources.first() else {
        polars_bail!(ComputeError: "expected at least 1 source");
    };

    let metadata = match first {
        ScanSourceRef::Path(addr) => match addr {
            PlPathRef::Cloud(uri) => {
                feature_gated!("cloud", {
                    let uri = uri.to_string();
                    get_runtime().block_on(async {
                        polars_io::ipc::IpcReaderAsync::from_uri(&uri, cloud_options)
                            .await?
                            .metadata()
                            .await
                    })?
                })
            },
            PlPathRef::Local(path) => arrow::io::ipc::read::read_file_metadata(
                &mut std::io::BufReader::new(polars_utils::open_file(path)?),
            )?,
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
        (None, 0),
    );

    Ok((file_info, metadata))
}

#[cfg(feature = "csv")]
pub fn csv_file_info(
    sources: &ScanSources,
    row_index: Option<&RowIndex>,
    csv_options: &mut CsvReadOptions,
    cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<FileInfo> {
    use std::io::{Read, Seek};

    use polars_core::error::feature_gated;
    use polars_core::{POOL, config};
    use polars_io::csv::read::schema_inference::SchemaInferenceResult;
    use polars_io::utils::get_reader_bytes;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    polars_ensure!(!sources.is_empty(), ComputeError: "expected at least 1 source");

    // TODO:
    // * See if we can do better than scanning all files if there is a row limit
    // * See if we can do this without downloading the entire file

    // prints the error message if paths is empty.
    let run_async = sources.is_cloud_url() || (sources.is_paths() && config::force_async());

    let cache_entries = {
        if run_async {
            feature_gated!("cloud", {
                Some(polars_io::file_cache::init_entries_from_uri_list(
                    sources
                        .as_paths()
                        .unwrap()
                        .iter()
                        .map(|path| Arc::from(path.to_str())),
                    cloud_options,
                )?)
            })
        } else {
            None
        }
    };

    let infer_schema_func = |i| {
        let source = sources.at(i);
        let memslice = source.to_memslice_possibly_async(run_async, cache_entries.as_ref(), i)?;
        let owned = &mut vec![];
        let mut reader = std::io::Cursor::new(maybe_decompress_bytes(&memslice, owned)?);
        if reader.read(&mut [0; 4])? < 2 && csv_options.raise_if_empty {
            polars_bail!(NoData: "empty CSV")
        }
        reader.rewind()?;

        let reader_bytes = get_reader_bytes(&mut reader).expect("could not mmap file");

        // this needs a way to estimated bytes/rows.
        SchemaInferenceResult::try_from_reader_bytes_and_options(&reader_bytes, csv_options)
    };

    let merge_func = |a: PolarsResult<SchemaInferenceResult>,
                      b: PolarsResult<SchemaInferenceResult>| {
        match (a, b) {
            (Err(e), _) | (_, Err(e)) => Err(e),
            (Ok(a), Ok(b)) => {
                let merged_schema = if csv_options.schema.is_some() {
                    csv_options.schema.clone().unwrap()
                } else {
                    let schema_a = a.get_inferred_schema();
                    let schema_b = b.get_inferred_schema();

                    match (schema_a.is_empty(), schema_b.is_empty()) {
                        (true, _) => schema_b,
                        (_, true) => schema_a,
                        _ => {
                            let mut s = Arc::unwrap_or_clone(schema_a);
                            s.to_supertype(&schema_b)?;
                            Arc::new(s)
                        },
                    }
                };

                Ok(a.with_inferred_schema(merged_schema))
            },
        }
    };

    let si_results = POOL.join(
        || infer_schema_func(0),
        || {
            (1..sources.len())
                .into_par_iter()
                .map(infer_schema_func)
                .reduce(|| Ok(Default::default()), merge_func)
        },
    );

    let si_result = merge_func(si_results.0, si_results.1)?;

    csv_options.update_with_inference_result(&si_result);

    let mut schema = csv_options
        .schema
        .clone()
        .unwrap_or_else(|| si_result.get_inferred_schema());

    let reader_schema = if let Some(rc) = row_index {
        let reader_schema = schema.clone();
        let mut output_schema = (*reader_schema).clone();
        insert_row_index_to_schema(&mut output_schema, rc.name.clone())?;
        schema = Arc::new(output_schema);
        reader_schema
    } else {
        schema.clone()
    };

    let estimated_n_rows = si_result.get_estimated_n_rows();

    Ok(FileInfo::new(
        schema,
        Some(Either::Right(reader_schema)),
        (None, estimated_n_rows),
    ))
}

#[cfg(feature = "json")]
pub fn ndjson_file_info(
    sources: &ScanSources,
    row_index: Option<&RowIndex>,
    ndjson_options: &NDJsonReadOptions,
    cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<FileInfo> {
    use polars_core::config;
    use polars_core::error::feature_gated;

    let Some(first) = sources.first() else {
        polars_bail!(ComputeError: "expected at least 1 source");
    };

    let run_async = sources.is_cloud_url() || (sources.is_paths() && config::force_async());

    let cache_entries = {
        if run_async {
            feature_gated!("cloud", {
                Some(polars_io::file_cache::init_entries_from_uri_list(
                    sources
                        .as_paths()
                        .unwrap()
                        .iter()
                        .map(|path| Arc::from(path.to_str())),
                    cloud_options,
                )?)
            })
        } else {
            None
        }
    };

    let owned = &mut vec![];

    let mut schema = if let Some(schema) = ndjson_options.schema.clone() {
        schema
    } else {
        let memslice = first.to_memslice_possibly_async(run_async, cache_entries.as_ref(), 0)?;
        let mut reader = std::io::Cursor::new(maybe_decompress_bytes(&memslice, owned)?);

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
        first_path: PlPath,
        schema_overwrite: Option<SchemaRef>,
    },
    CsvJson {
        paths: Arc<[PlPath]>,
        schema: Option<SchemaRef>,
        schema_overwrite: Option<SchemaRef>,
    },
}

#[derive(Default)]
pub(super) struct SourcesToFileInfo {
    inner: PlHashMap<CachedSourceKey, (FileInfo, FileScanIR)>,
}

impl SourcesToFileInfo {
    fn infer_or_parse(
        &mut self,
        scan_type: FileScanDsl,
        sources: &ScanSources,
        unified_scan_args: &mut UnifiedScanArgs,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<(FileInfo, FileScanIR)> {
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
                            row_estimation: (None, 0),
                        },
                        FileScanIR::Parquet {
                            options,
                            metadata: None,
                        },
                    )
                } else {
                    let (file_info, metadata) = scans::parquet_file_info(
                        sources,
                        unified_scan_args.row_index.as_ref(),
                        cloud_options,
                    )
                    .map_err(|e| e.context(failed_here!(parquet scan)))?;

                    (file_info, FileScanIR::Parquet { options, metadata })
                }
            },
            #[cfg(feature = "ipc")]
            FileScanDsl::Ipc { options } => {
                let (file_info, md) = scans::ipc_file_info(
                    sources,
                    unified_scan_args.row_index.as_ref(),
                    cloud_options,
                )
                .map_err(|e| e.context(failed_here!(ipc scan)))?;
                (
                    file_info,
                    FileScanIR::Ipc {
                        options,
                        metadata: Some(Arc::new(md)),
                    },
                )
            },
            #[cfg(feature = "csv")]
            FileScanDsl::Csv { mut options } => {
                // TODO: This is a hack. We conditionally set `allow_missing_columns` to
                // mimic existing behavior, but this should be taken from a user provided
                // parameter instead.
                if options.schema.is_some() && options.has_header {
                    unified_scan_args.missing_columns_policy = MissingColumnsPolicy::Insert;
                }

                (
                    scans::csv_file_info(
                        sources,
                        unified_scan_args.row_index.as_ref(),
                        &mut options,
                        cloud_options,
                    )
                    .map_err(|e| e.context(failed_here!(csv scan)))?,
                    FileScanIR::Csv { options },
                )
            },
            #[cfg(feature = "json")]
            FileScanDsl::NDJson { options } => (
                scans::ndjson_file_info(
                    sources,
                    unified_scan_args.row_index.as_ref(),
                    &options,
                    cloud_options,
                )
                .map_err(|e| e.context(failed_here!(ndjson scan)))?,
                FileScanIR::NDJson { options },
            ),
            #[cfg(feature = "python")]
            FileScanDsl::PythonDataset { dataset_object } => {
                if crate::dsl::DATASET_PROVIDER_VTABLE.get().is_none() {
                    polars_bail!(ComputeError: "DATASET_PROVIDER_VTABLE (python) not initialized")
                }

                let mut schema = dataset_object.schema()?;
                let reader_schema = schema.clone();

                if let Some(row_index) = &unified_scan_args.row_index {
                    insert_row_index_to_schema(Arc::make_mut(&mut schema), row_index.name.clone())?;
                }

                (
                    FileInfo {
                        schema,
                        reader_schema: Some(either::Either::Right(reader_schema)),
                        row_estimation: (None, usize::MAX),
                    },
                    FileScanIR::PythonDataset {
                        dataset_object,
                        cached_ir: Default::default(),
                    },
                )
            },
            FileScanDsl::Anonymous {
                file_info,
                options,
                function,
            } => (file_info, FileScanIR::Anonymous { options, function }),
        })
    }

    pub(super) fn get_or_insert(
        &mut self,
        scan_type: &FileScanDsl,
        sources: &ScanSources,
        unified_scan_args: &mut UnifiedScanArgs,
        cloud_options: Option<&CloudOptions>,
        verbose: bool,
    ) -> PolarsResult<(FileInfo, FileScanIR)> {
        // Only cache paths. Others are directly parsed.
        let ScanSources::Paths(paths) = sources else {
            return self.infer_or_parse(
                scan_type.clone(),
                sources,
                unified_scan_args,
                cloud_options,
            );
        };
        if paths.is_empty() {
            return self.infer_or_parse(
                scan_type.clone(),
                sources,
                unified_scan_args,
                cloud_options,
            );
        }

        let (k, v): (CachedSourceKey, Option<&(FileInfo, FileScanIR)>) = match scan_type {
            #[cfg(feature = "parquet")]
            FileScanDsl::Parquet { options } => {
                let key = CachedSourceKey::ParquetIpc {
                    first_path: paths[0].clone(),
                    schema_overwrite: options.schema.clone(),
                };

                let v = self.inner.get(&key);
                (key, v)
            },
            #[cfg(feature = "ipc")]
            FileScanDsl::Ipc { options: _ } => {
                let key = CachedSourceKey::ParquetIpc {
                    first_path: paths[0].clone(),
                    schema_overwrite: None,
                };

                let v = self.inner.get(&key);
                (key, v)
            },
            #[cfg(feature = "csv")]
            FileScanDsl::Csv { options } => {
                let key = CachedSourceKey::CsvJson {
                    paths: paths.clone(),
                    schema: options.schema.clone(),
                    schema_overwrite: options.schema_overwrite.clone(),
                };
                let v = self.inner.get(&key);
                (key, v)
            },
            #[cfg(feature = "json")]
            FileScanDsl::NDJson { options } => {
                let key = CachedSourceKey::CsvJson {
                    paths: paths.clone(),
                    schema: options.schema.clone(),
                    schema_overwrite: options.schema_overwrite.clone(),
                };
                let v = self.inner.get(&key);
                (key, v)
            },
            _ => {
                return self.infer_or_parse(
                    scan_type.clone(),
                    sources,
                    unified_scan_args,
                    cloud_options,
                );
            },
        };

        if let Some(out) = v {
            if verbose {
                eprintln!("FILE_INFO CACHE HIT")
            }
            Ok(out.clone())
        } else {
            let v =
                self.infer_or_parse(scan_type.clone(), sources, unified_scan_args, cloud_options)?;
            self.inner.insert(k, v.clone());
            Ok(v)
        }
    }
}
