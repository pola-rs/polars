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
                ScanSources::Paths(Arc::from(["dummy".into()]))
            },
            FileScanDsl::Anonymous { .. } => sources,
        };

        // For cloud we must deduplicate files. Serialization/deserialization leads to Arc's losing there
        // sharing.
        // First we check if we have a cache `FileInfo`, if not we load the metadata and
        // insert in the cache. Loading metadata can be expensive, and comprise of multiple
        // Gb's
        let (mut file_info, scan_type_ir) = if let Some(file_info) =
            ctxt.cache_file_info.get(&sources)
        {
            if ctxt.verbose {
                eprintln!("FILE_INFO CACHE HIT")
            }
            let scan_type_ir = match *scan_type.clone() {
                #[cfg(feature = "csv")]
                FileScanDsl::Csv { options } => FileScanIR::Csv { options },
                #[cfg(feature = "json")]
                FileScanDsl::NDJson { options } => FileScanIR::NDJson { options },
                #[cfg(feature = "parquet")]
                FileScanDsl::Parquet { options } => FileScanIR::Parquet {
                    options,
                    metadata: None,
                },
                #[cfg(feature = "ipc")]
                FileScanDsl::Ipc { options } => FileScanIR::Ipc {
                    options,
                    metadata: None,
                },
                #[cfg(feature = "python")]
                FileScanDsl::PythonDataset { dataset_object } => FileScanIR::PythonDataset {
                    dataset_object,
                    cached_ir: Default::default(),
                },
                FileScanDsl::Anonymous {
                    options,
                    function,
                    file_info: _,
                } => FileScanIR::Anonymous { options, function },
            };
            (file_info.clone(), scan_type_ir)
        } else {
            let (file_info, scan_type_ir) = match *scan_type.clone() {
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
                            &sources,
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
                        &sources,
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
                            &sources,
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
                        &sources,
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
                        insert_row_index_to_schema(
                            Arc::make_mut(&mut schema),
                            row_index.name.clone(),
                        )?;
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
            };

            // Insert so that the files are deduplicated.
            ctxt.cache_file_info.insert(&sources, &file_info);
            (file_info, scan_type_ir)
        };

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
                id: Default::default(),
            }
        };

        cached_ir.replace(ir);
    }

    Ok(cached_ir.clone().unwrap())
}
