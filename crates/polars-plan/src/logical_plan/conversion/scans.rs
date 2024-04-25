use std::io::Read;
use std::path::PathBuf;

#[cfg(feature = "cloud")]
use polars_io::pl_async::get_runtime;
use polars_io::prelude::*;
use polars_io::utils::is_cloud_url;
use polars_io::RowIndex;

use super::*;

fn get_path(paths: &[PathBuf]) -> PolarsResult<&PathBuf> {
    // Use first path to get schema.
    paths
        .first()
        .ok_or_else(|| polars_err!(ComputeError: "expected at least 1 path"))
}

#[cfg(any(feature = "parquet", feature = "parquet_async",))]
fn prepare_schema(mut schema: Schema, row_index: Option<&RowIndex>) -> SchemaRef {
    if let Some(rc) = row_index {
        let _ = schema.insert_at_index(0, rc.name.as_str().into(), IDX_DTYPE);
    }
    Arc::new(schema)
}

#[cfg(feature = "parquet")]
pub(super) fn parquet_file_info(
    paths: &[PathBuf],
    file_options: &FileScanOptions,
    cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<(FileInfo, Option<FileMetaDataRef>)> {
    let path = get_path(paths)?;

    let (schema, reader_schema, num_rows, metadata) = if is_cloud_url(path) {
        #[cfg(not(feature = "cloud"))]
        panic!("One or more of the cloud storage features ('aws', 'gcp', ...) must be enabled.");

        #[cfg(feature = "cloud")]
        {
            let uri = path.to_string_lossy();
            get_runtime().block_on(async {
                let mut reader =
                    ParquetAsyncReader::from_uri(&uri, cloud_options, None, None).await?;
                let reader_schema = reader.schema().await?;
                let num_rows = reader.num_rows().await?;
                let metadata = reader.get_metadata().await?.clone();

                let schema =
                    prepare_schema((&reader_schema).into(), file_options.row_index.as_ref());
                PolarsResult::Ok((schema, reader_schema, Some(num_rows), Some(metadata)))
            })?
        }
    } else {
        let file = polars_utils::open_file(path)?;
        let mut reader = ParquetReader::new(file);
        let reader_schema = reader.schema()?;
        let schema = prepare_schema((&reader_schema).into(), file_options.row_index.as_ref());
        (
            schema,
            reader_schema,
            Some(reader.num_rows()?),
            Some(reader.get_metadata()?.clone()),
        )
    };

    let mut file_info = FileInfo::new(
        schema,
        Some(reader_schema),
        (num_rows, num_rows.unwrap_or(0)),
    );

    if file_options.hive_options.enabled {
        file_info.init_hive_partitions(path.as_path(), file_options.hive_options.schema.clone())?
    }

    Ok((file_info, metadata))
}

// TODO! return metadata arced
#[cfg(feature = "ipc")]
pub(super) fn ipc_file_info(
    paths: &[PathBuf],
    file_options: &FileScanOptions,
    cloud_options: Option<&polars_io::cloud::CloudOptions>,
) -> PolarsResult<(FileInfo, arrow::io::ipc::read::FileMetadata)> {
    let path = get_path(paths)?;

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
        prepare_schema(
            metadata.schema.as_ref().into(),
            file_options.row_index.as_ref(),
        ),
        Some(Arc::clone(&metadata.schema)),
        (None, 0),
    );

    Ok((file_info, metadata))
}

#[cfg(feature = "csv")]
pub(super) fn csv_file_info(
    paths: &[PathBuf],
    file_options: &FileScanOptions,
    csv_options: &mut CsvReaderOptions,
) -> PolarsResult<FileInfo> {
    use std::io::Seek;

    use polars_io::csv::read::{infer_file_schema, is_compressed};
    use polars_io::utils::get_reader_bytes;

    let path = get_path(paths)?;
    let mut file = polars_utils::open_file(path)?;

    let mut magic_nr = [0u8; 4];
    let res_len = file.read(&mut magic_nr)?;
    if res_len < 2 {
        if csv_options.raise_if_empty {
            polars_bail!(NoData: "empty CSV")
        }
    } else {
        polars_ensure!(
        !is_compressed(&magic_nr),
        ComputeError: "cannot scan compressed csv; use `read_csv` for compressed data",
        );
    }

    file.rewind()?;
    let reader_bytes = get_reader_bytes(&mut file).expect("could not mmap file");

    // this needs a way to estimated bytes/rows.
    let (inferred_schema, rows_read, bytes_read) = infer_file_schema(
        &reader_bytes,
        csv_options.separator,
        csv_options.infer_schema_length,
        csv_options.has_header,
        csv_options.schema_overwrite.as_deref(),
        &mut csv_options.skip_rows,
        csv_options.skip_rows_after_header,
        csv_options.comment_prefix.as_ref(),
        csv_options.quote_char,
        csv_options.eol_char,
        csv_options.null_values.as_ref(),
        csv_options.try_parse_dates,
        csv_options.raise_if_empty,
        &mut csv_options.n_threads,
        csv_options.decimal_comma,
    )?;

    let mut schema = csv_options
        .schema
        .clone()
        .unwrap_or_else(|| Arc::new(inferred_schema));

    if let Some(rc) = &file_options.row_index {
        let schema = Arc::make_mut(&mut schema);
        schema.insert_at_index(0, rc.name.as_str().into(), IDX_DTYPE)?;
    }

    let n_bytes = reader_bytes.len();
    let estimated_n_rows = (rows_read as f64 / bytes_read as f64 * n_bytes as f64) as usize;

    csv_options.skip_rows += csv_options.skip_rows_after_header;
    Ok(FileInfo::new(schema, None, (None, estimated_n_rows)))
}
