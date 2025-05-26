#[cfg(feature = "ipc")]
use arrow::io::ipc::read::get_row_count as count_rows_ipc_sync;
#[cfg(any(
    feature = "parquet",
    feature = "ipc",
    feature = "json",
    feature = "csv"
))]
use polars_core::error::feature_gated;
#[cfg(any(feature = "json", feature = "parquet"))]
use polars_io::SerReader;
#[cfg(any(feature = "parquet", feature = "json"))]
use polars_io::cloud::CloudOptions;
#[cfg(feature = "parquet")]
use polars_io::parquet::read::ParquetReader;
#[cfg(all(feature = "parquet", feature = "async"))]
use polars_io::pl_async::{get_runtime, with_concurrency_budget};

use super::*;

#[allow(unused_variables)]
pub fn count_rows(
    sources: &ScanSources,
    scan_type: &FileScan,
    cloud_options: Option<&CloudOptions>,
    alias: Option<PlSmallStr>,
) -> PolarsResult<DataFrame> {
    #[cfg(not(any(
        feature = "parquet",
        feature = "ipc",
        feature = "json",
        feature = "csv"
    )))]
    {
        unreachable!()
    }

    #[cfg(any(
        feature = "parquet",
        feature = "ipc",
        feature = "json",
        feature = "csv"
    ))]
    {
        let count: PolarsResult<usize> = match scan_type {
            #[cfg(feature = "csv")]
            FileScan::Csv { options } => count_all_rows_csv(sources, options),
            #[cfg(feature = "parquet")]
            FileScan::Parquet { .. } => count_rows_parquet(sources, cloud_options),
            #[cfg(feature = "ipc")]
            FileScan::Ipc { options, metadata } => count_rows_ipc(
                sources,
                #[cfg(feature = "cloud")]
                cloud_options,
                metadata.as_deref(),
            ),
            #[cfg(feature = "json")]
            FileScan::NDJson { options } => count_rows_ndjson(sources, cloud_options),
            #[cfg(feature = "python")]
            FileScan::PythonDataset { .. } => unreachable!(),
            FileScan::Anonymous { .. } => {
                unreachable!()
            },
        };
        let count = count?;
        let count: IdxSize = count.try_into().map_err(
            |_| polars_err!(ComputeError: "count of {} exceeded maximum row size", count),
        )?;
        let column_name = alias.unwrap_or(PlSmallStr::from_static(crate::constants::LEN));
        DataFrame::new(vec![Column::new(column_name, [count])])
    }
}

#[cfg(feature = "csv")]
fn count_all_rows_csv(
    sources: &ScanSources,
    options: &polars_io::prelude::CsvReadOptions,
) -> PolarsResult<usize> {
    let parse_options = options.get_parse_options();

    sources
        .iter()
        .map(|source| match source {
            ScanSourceRef::Path(path) => polars_io::csv::read::count_rows(
                path,
                parse_options.separator,
                parse_options.quote_char,
                parse_options.comment_prefix.as_ref(),
                parse_options.eol_char,
                options.has_header,
            ),
            _ => {
                let memslice = source.to_memslice()?;

                polars_io::csv::read::count_rows_from_slice_par(
                    &memslice[..],
                    parse_options.separator,
                    parse_options.quote_char,
                    parse_options.comment_prefix.as_ref(),
                    parse_options.eol_char,
                    options.has_header,
                )
            },
        })
        .sum()
}

#[cfg(feature = "parquet")]
pub(super) fn count_rows_parquet(
    sources: &ScanSources,
    #[allow(unused)] cloud_options: Option<&CloudOptions>,
) -> PolarsResult<usize> {
    if sources.is_empty() {
        return Ok(0);
    };
    let is_cloud = sources.is_cloud_url();

    if is_cloud {
        feature_gated!("cloud", {
            get_runtime().block_on(count_rows_cloud_parquet(
                sources.as_paths().unwrap(),
                cloud_options,
            ))
        })
    } else {
        sources
            .iter()
            .map(|source| {
                ParquetReader::new(std::io::Cursor::new(source.to_memslice()?)).num_rows()
            })
            .sum::<PolarsResult<usize>>()
    }
}

#[cfg(all(feature = "parquet", feature = "async"))]
async fn count_rows_cloud_parquet(
    paths: &[std::path::PathBuf],
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<usize> {
    use polars_io::prelude::ParquetObjectStore;

    let collection = paths.iter().map(|path| {
        with_concurrency_budget(1, || async {
            let mut reader =
                ParquetObjectStore::from_uri(&path.to_string_lossy(), cloud_options, None).await?;
            reader.num_rows().await
        })
    });
    futures::future::try_join_all(collection)
        .await
        .map(|rows| rows.iter().sum())
}

#[cfg(feature = "ipc")]
pub(super) fn count_rows_ipc(
    sources: &ScanSources,
    #[cfg(feature = "cloud")] cloud_options: Option<&CloudOptions>,
    metadata: Option<&arrow::io::ipc::read::FileMetadata>,
) -> PolarsResult<usize> {
    if sources.is_empty() {
        return Ok(0);
    };
    let is_cloud = sources.is_cloud_url();

    if is_cloud {
        feature_gated!("cloud", {
            get_runtime().block_on(count_rows_cloud_ipc(
                sources.as_paths().unwrap(),
                cloud_options,
                metadata,
            ))
        })
    } else {
        sources
            .iter()
            .map(|source| {
                let memslice = source.to_memslice()?;
                count_rows_ipc_sync(&mut std::io::Cursor::new(memslice)).map(|v| v as usize)
            })
            .sum::<PolarsResult<usize>>()
    }
}

#[cfg(all(feature = "ipc", feature = "async"))]
async fn count_rows_cloud_ipc(
    paths: &[std::path::PathBuf],
    cloud_options: Option<&CloudOptions>,
    metadata: Option<&arrow::io::ipc::read::FileMetadata>,
) -> PolarsResult<usize> {
    use polars_io::ipc::IpcReaderAsync;

    let collection = paths.iter().map(|path| {
        with_concurrency_budget(1, || async {
            let reader = IpcReaderAsync::from_uri(&path.to_string_lossy(), cloud_options).await?;
            reader.count_rows(metadata).await
        })
    });
    futures::future::try_join_all(collection)
        .await
        .map(|rows| rows.iter().map(|v| *v as usize).sum())
}

#[cfg(feature = "json")]
pub(super) fn count_rows_ndjson(
    sources: &ScanSources,
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<usize> {
    use polars_core::config;
    use polars_io::utils::compression::maybe_decompress_bytes;

    if sources.is_empty() {
        return Ok(0);
    }

    let is_cloud_url = sources.is_cloud_url();
    let run_async = is_cloud_url || (sources.is_paths() && config::force_async());

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

    sources
        .iter()
        .map(|source| {
            let memslice =
                source.to_memslice_possibly_async(run_async, cache_entries.as_ref(), 0)?;

            let owned = &mut vec![];
            let reader = polars_io::ndjson::core::JsonLineReader::new(std::io::Cursor::new(
                maybe_decompress_bytes(&memslice[..], owned)?,
            ));
            reader.count()
        })
        .sum()
}
