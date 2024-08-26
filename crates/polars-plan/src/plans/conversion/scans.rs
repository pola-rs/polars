use std::path::PathBuf;

use either::Either;
use polars_io::path_utils::is_cloud_url;
#[cfg(feature = "cloud")]
use polars_io::pl_async::get_runtime;
use polars_io::prelude::*;
use polars_io::RowIndex;

use super::*;

fn get_first_path(paths: &[PathBuf]) -> PolarsResult<&PathBuf> {
    // Use first path to get schema.
    paths
        .first()
        .ok_or_else(|| polars_err!(ComputeError: "expected at least 1 path"))
}

#[cfg(any(feature = "parquet", feature = "ipc"))]
fn prepare_output_schema(mut schema: Schema, row_index: Option<&RowIndex>) -> SchemaRef {
    if let Some(rc) = row_index {
        let _ = schema.insert_at_index(0, rc.name.as_ref().into(), IDX_DTYPE);
    }
    Arc::new(schema)
}

#[cfg(any(feature = "json", feature = "csv"))]
fn prepare_schemas(mut schema: Schema, row_index: Option<&RowIndex>) -> (SchemaRef, SchemaRef) {
    if let Some(rc) = row_index {
        let reader_schema = schema.clone();
        let _ = schema.insert_at_index(0, rc.name.as_ref().into(), IDX_DTYPE);
        (Arc::new(reader_schema), Arc::new(schema))
    } else {
        let schema = Arc::new(schema);
        (schema.clone(), schema)
    }
}

#[cfg(feature = "parquet")]
pub(super) fn parquet_file_info(
    paths: &[PathBuf],
    file_options: &FileScanOptions,
    #[allow(unused)] cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<(FileInfo, Option<FileMetaDataRef>)> {
    let path = get_first_path(paths)?;

    let (schema, reader_schema, num_rows, metadata) = if is_cloud_url(path) {
        #[cfg(not(feature = "cloud"))]
        panic!("One or more of the cloud storage features ('aws', 'gcp', ...) must be enabled.");

        #[cfg(feature = "cloud")]
        {
            let uri = path.to_string_lossy();
            get_runtime().block_on(async {
                let mut reader = ParquetAsyncReader::from_uri(&uri, cloud_options, None).await?;
                let reader_schema = reader.schema().await?;
                let num_rows = reader.num_rows().await?;
                let metadata = reader.get_metadata().await?.clone();

                let schema =
                    prepare_output_schema((&reader_schema).into(), file_options.row_index.as_ref());
                PolarsResult::Ok((schema, reader_schema, Some(num_rows), Some(metadata)))
            })?
        }
    } else {
        let file = polars_utils::open_file(path)?;
        let mut reader = ParquetReader::new(file);
        let reader_schema = reader.schema()?;
        let schema =
            prepare_output_schema((&reader_schema).into(), file_options.row_index.as_ref());
        (
            schema,
            reader_schema,
            Some(reader.num_rows()?),
            Some(reader.get_metadata()?.clone()),
        )
    };

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
    paths: &[PathBuf],
    file_options: &FileScanOptions,
    cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<(FileInfo, arrow::io::ipc::read::FileMetadata)> {
    let path = get_first_path(paths)?;

    let metadata = if is_cloud_url(path) {
        #[cfg(not(feature = "cloud"))]
        panic!("One or more of the cloud storage features ('aws', 'gcp', ...) must be enabled.");

        #[cfg(feature = "cloud")]
        {
            let uri = path.to_string_lossy();
            get_runtime().block_on(async {
                polars_io::ipc::IpcReaderAsync::from_uri(&uri, cloud_options)
                    .await?
                    .metadata()
                    .await
            })?
        }
    } else {
        arrow::io::ipc::read::read_file_metadata(&mut std::io::BufReader::new(
            polars_utils::open_file(path)?,
        ))?
    };
    let file_info = FileInfo::new(
        prepare_output_schema(
            metadata.schema.as_ref().into(),
            file_options.row_index.as_ref(),
        ),
        Some(Either::Left(Arc::clone(&metadata.schema))),
        (None, 0),
    );

    Ok((file_info, metadata))
}

#[cfg(feature = "csv")]
pub(super) fn csv_file_info(
    paths: &[PathBuf],
    file_options: &FileScanOptions,
    csv_options: &mut CsvReadOptions,
    cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<FileInfo> {
    use std::io::{Read, Seek};

    use polars_core::{config, POOL};
    use polars_io::csv::read::schema_inference::SchemaInferenceResult;
    use polars_io::utils::get_reader_bytes;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    // TODO:
    // * See if we can do better than scanning all files if there is a row limit
    // * See if we can do this without downloading the entire file

    // prints the error message if paths is empty.
    let first_path = get_first_path(paths)?;
    let run_async = is_cloud_url(first_path) || config::force_async();

    let cache_entries = {
        #[cfg(feature = "cloud")]
        {
            if run_async {
                Some(polars_io::file_cache::init_entries_from_uri_list(
                    paths
                        .iter()
                        .map(|path| Arc::from(path.to_str().unwrap()))
                        .collect::<Vec<_>>()
                        .as_slice(),
                    cloud_options,
                )?)
            } else {
                None
            }
        }
        #[cfg(not(feature = "cloud"))]
        {
            if run_async {
                panic!("required feature `cloud` is not enabled")
            }
        }
    };

    let infer_schema_func = |i| {
        let file = if run_async {
            #[cfg(feature = "cloud")]
            {
                let entry: &Arc<polars_io::file_cache::FileCacheEntry> =
                    &cache_entries.as_ref().unwrap()[i];
                entry.try_open_check_latest()?
            }
            #[cfg(not(feature = "cloud"))]
            {
                panic!("required feature `cloud` is not enabled")
            }
        } else {
            let p: &PathBuf = &paths[i];
            polars_utils::open_file(p.as_ref())?
        };

        let mmap = unsafe { memmap::Mmap::map(&file).unwrap() };
        let owned = &mut vec![];

        let mut curs = std::io::Cursor::new(maybe_decompress_bytes(mmap.as_ref(), owned)?);

        if curs.read(&mut [0; 4])? < 2 && csv_options.raise_if_empty {
            polars_bail!(NoData: "empty CSV")
        }
        curs.rewind()?;

        let reader_bytes = get_reader_bytes(&mut curs).expect("could not mmap file");

        // this needs a way to estimated bytes/rows.
        let si_result =
            SchemaInferenceResult::try_from_reader_bytes_and_options(&reader_bytes, csv_options)?;

        Ok(si_result)
    };

    let merge_func = |a: PolarsResult<SchemaInferenceResult>,
                      b: PolarsResult<SchemaInferenceResult>| match (a, b) {
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
    };

    let si_results = POOL.join(
        || infer_schema_func(0),
        || {
            (1..paths.len())
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

    let reader_schema = if let Some(rc) = &file_options.row_index {
        let reader_schema = schema.clone();
        let mut output_schema = (*reader_schema).clone();
        output_schema.insert_at_index(0, rc.name.as_ref().into(), IDX_DTYPE)?;
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
pub(super) fn ndjson_file_info(
    paths: &[PathBuf],
    file_options: &FileScanOptions,
    ndjson_options: &mut NDJsonReadOptions,
    cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<FileInfo> {
    use polars_core::config;

    let run_async = !paths.is_empty() && is_cloud_url(&paths[0]) || config::force_async();

    let cache_entries = {
        #[cfg(feature = "cloud")]
        {
            if run_async {
                Some(polars_io::file_cache::init_entries_from_uri_list(
                    paths
                        .iter()
                        .map(|path| Arc::from(path.to_str().unwrap()))
                        .collect::<Vec<_>>()
                        .as_slice(),
                    cloud_options,
                )?)
            } else {
                None
            }
        }
        #[cfg(not(feature = "cloud"))]
        {
            if run_async {
                panic!("required feature `cloud` is not enabled")
            }
        }
    };

    let first_path = get_first_path(paths)?;

    let f = if run_async {
        #[cfg(feature = "cloud")]
        {
            cache_entries.unwrap()[0].try_open_check_latest()?
        }
        #[cfg(not(feature = "cloud"))]
        {
            panic!("required feature `cloud` is not enabled")
        }
    } else {
        polars_utils::open_file(first_path)?
    };

    let owned = &mut vec![];
    let mmap = unsafe { memmap::Mmap::map(&f).unwrap() };

    let mut reader = std::io::BufReader::new(maybe_decompress_bytes(mmap.as_ref(), owned)?);

    let (mut reader_schema, schema) = if let Some(schema) = ndjson_options.schema.take() {
        if file_options.row_index.is_none() {
            (schema.clone(), schema.clone())
        } else {
            prepare_schemas(
                Arc::unwrap_or_clone(schema),
                file_options.row_index.as_ref(),
            )
        }
    } else {
        let schema =
            polars_io::ndjson::infer_schema(&mut reader, ndjson_options.infer_schema_length)?;
        prepare_schemas(schema, file_options.row_index.as_ref())
    };

    if let Some(overwriting_schema) = &ndjson_options.schema_overwrite {
        let schema = Arc::make_mut(&mut reader_schema);
        overwrite_schema(schema, overwriting_schema)?;
    }

    Ok(FileInfo::new(
        schema,
        Some(Either::Right(reader_schema)),
        (None, usize::MAX),
    ))
}
