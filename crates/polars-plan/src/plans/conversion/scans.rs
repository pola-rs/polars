use either::Either;
use polars_io::RowIndex;
use polars_io::path_utils::is_cloud_url;
#[cfg(feature = "cloud")]
use polars_io::pl_async::get_runtime;
use polars_io::prelude::*;
use polars_io::utils::compression::maybe_decompress_bytes;

use super::*;

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
            let first_path = &sources.as_paths().unwrap()[0];
            feature_gated!("cloud", {
                let uri = first_path.to_string_lossy();
                get_runtime().block_in_place_on(async {
                    let mut reader =
                        ParquetObjectStore::from_uri(&uri, cloud_options, None).await?;

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

    let Some(first) = sources.first() else {
        polars_bail!(ComputeError: "expected at least 1 source");
    };

    let metadata = match first {
        ScanSourceRef::Path(path) => {
            if is_cloud_url(path) {
                feature_gated!("cloud", {
                    let uri = path.to_string_lossy();
                    get_runtime().block_on(async {
                        polars_io::ipc::IpcReaderAsync::from_uri(&uri, cloud_options)
                            .await?
                            .metadata()
                            .await
                    })?
                })
            } else {
                arrow::io::ipc::read::read_file_metadata(&mut std::io::BufReader::new(
                    polars_utils::open_file(path)?,
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
        (None, 0),
    );

    Ok((file_info, metadata))
}

#[cfg(feature = "csv")]
pub fn isolated_csv_file_info(
    source: ScanSourceRef,
    row_index: Option<&RowIndex>,
    csv_options: &mut CsvReadOptions,
    _cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<FileInfo> {
    use std::io::{Read, Seek};

    use polars_io::csv::read::schema_inference::SchemaInferenceResult;
    use polars_io::utils::get_reader_bytes;

    let run_async = source.run_async();

    let memslice = source.to_memslice_async_assume_latest(run_async)?;
    let owned = &mut vec![];
    let mut reader = std::io::Cursor::new(maybe_decompress_bytes(&memslice, owned)?);
    if reader.read(&mut [0; 4])? < 2 && csv_options.raise_if_empty {
        polars_bail!(NoData: "empty CSV")
    }
    reader.rewind()?;

    let reader_bytes = get_reader_bytes(&mut reader).expect("could not mmap file");

    // this needs a way to estimated bytes/rows.
    let si_result =
        SchemaInferenceResult::try_from_reader_bytes_and_options(&reader_bytes, csv_options)?;

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
                        .map(|path| Arc::from(path.to_str().unwrap()))
                        .collect::<Vec<_>>()
                        .as_slice(),
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
                        .map(|path| Arc::from(path.to_str().unwrap()))
                        .collect::<Vec<_>>()
                        .as_slice(),
                    cloud_options,
                )?)
            })
        } else {
            None
        }
    };

    let owned = &mut vec![];

    let mut schema = if let Some(schema) = ndjson_options.schema.clone() {
        schema.clone()
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
