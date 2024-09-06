#[cfg(feature = "ipc")]
use arrow::io::ipc::read::get_row_count as count_rows_ipc_sync;
#[cfg(any(
    feature = "parquet",
    feature = "ipc",
    feature = "json",
    feature = "csv"
))]
use polars_core::error::feature_gated;
#[cfg(any(feature = "parquet", feature = "json"))]
use polars_io::cloud::CloudOptions;
#[cfg(feature = "csv")]
use polars_io::csv::read::{
    count_rows as count_rows_csv, count_rows_from_slice as count_rows_csv_from_slice,
};
#[cfg(all(feature = "parquet", feature = "cloud"))]
use polars_io::parquet::read::ParquetAsyncReader;
#[cfg(feature = "parquet")]
use polars_io::parquet::read::ParquetReader;
#[cfg(all(feature = "parquet", feature = "async"))]
use polars_io::pl_async::{get_runtime, with_concurrency_budget};
#[cfg(any(feature = "json", feature = "parquet"))]
use polars_io::SerReader;

use super::*;

#[allow(unused_variables)]
pub fn count_rows(sources: &ScanSources, scan_type: &FileScan) -> PolarsResult<DataFrame> {
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
            FileScan::Csv {
                options,
                cloud_options,
            } => count_all_rows_csv(sources, options),
            #[cfg(feature = "parquet")]
            FileScan::Parquet { cloud_options, .. } => {
                count_rows_parquet(sources, cloud_options.as_ref())
            },
            #[cfg(feature = "ipc")]
            FileScan::Ipc {
                options,
                cloud_options,
                metadata,
            } => count_rows_ipc(
                sources,
                #[cfg(feature = "cloud")]
                cloud_options.as_ref(),
                metadata.as_ref(),
            ),
            #[cfg(feature = "json")]
            FileScan::NDJson {
                options,
                cloud_options,
            } => count_rows_ndjson(sources, cloud_options.as_ref()),
            FileScan::Anonymous { .. } => {
                unreachable!()
            },
        };
        let count = count?;
        let count: IdxSize = count.try_into().map_err(
            |_| polars_err!(ComputeError: "count of {} exceeded maximum row size", count),
        )?;
        DataFrame::new(vec![Series::new(
            PlSmallStr::from_static(crate::constants::LEN),
            [count],
        )])
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
            ScanSourceRef::File(path) => count_rows_csv(
                path,
                parse_options.separator,
                parse_options.quote_char,
                parse_options.comment_prefix.as_ref(),
                parse_options.eol_char,
                options.has_header,
            ),
            ScanSourceRef::Buffer(buf) => count_rows_csv_from_slice(
                &buf[..],
                parse_options.separator,
                parse_options.quote_char,
                parse_options.comment_prefix.as_ref(),
                parse_options.eol_char,
                options.has_header,
            ),
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
                sources.as_paths().ok_or_else(|| {
                    polars_err!(nyi = "Asynchronous scanning of in-memory buffers")
                })?,
                cloud_options,
            ))
        })
    } else {
        sources
            .iter()
            .map(|source| match source {
                ScanSourceRef::File(path) => {
                    ParquetReader::new(polars_utils::open_file(path)?).num_rows()
                },
                ScanSourceRef::Buffer(buffer) => {
                    ParquetReader::new(std::io::Cursor::new(buffer)).num_rows()
                },
            })
            .sum::<PolarsResult<usize>>()
    }
}

#[cfg(all(feature = "parquet", feature = "async"))]
async fn count_rows_cloud_parquet(
    paths: &[std::path::PathBuf],
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<usize> {
    let collection = paths.iter().map(|path| {
        with_concurrency_budget(1, || async {
            let mut reader =
                ParquetAsyncReader::from_uri(&path.to_string_lossy(), cloud_options, None).await?;
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
                sources.as_paths().ok_or_else(|| {
                    polars_err!(nyi = "Asynchronous scanning of in-memory buffers")
                })?,
                cloud_options,
                metadata,
            ))
        })
    } else {
        sources
            .iter()
            .map(|source| match source {
                ScanSourceRef::File(path) => {
                    count_rows_ipc_sync(&mut polars_utils::open_file(path)?).map(|v| v as usize)
                },
                ScanSourceRef::Buffer(buffer) => {
                    count_rows_ipc_sync(&mut std::io::Cursor::new(buffer)).map(|v| v as usize)
                },
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
    use polars_io::utils::maybe_decompress_bytes;

    if sources.is_empty() {
        return Ok(0);
    }

    let is_cloud_url = sources.is_cloud_url();
    let run_async = is_cloud_url || config::force_async();

    let cache_entries = {
        feature_gated!("cloud", {
            if run_async {
                Some(polars_io::file_cache::init_entries_from_uri_list(
                    sources
                        .as_paths()
                        .ok_or_else(|| {
                            polars_err!(nyi = "Asynchronous scanning of in-memory buffers")
                        })?
                        .iter()
                        .map(|path| Arc::from(path.to_str().unwrap()))
                        .collect::<Vec<_>>()
                        .as_slice(),
                    cloud_options,
                )?)
            } else {
                None
            }
        })
    };

    sources
        .iter()
        .map(|source| match source {
            ScanSourceRef::File(path) => {
                let f = if run_async {
                    feature_gated!("cloud", {
                        let entry: &Arc<polars_io::file_cache::FileCacheEntry> =
                            &cache_entries.as_ref().unwrap()[0];
                        entry.try_open_check_latest()?
                    })
                } else {
                    polars_utils::open_file(path)?
                };

                let mmap = unsafe { memmap::Mmap::map(&f).unwrap() };
                let owned = &mut vec![];

                let reader = polars_io::ndjson::core::JsonLineReader::new(std::io::Cursor::new(
                    maybe_decompress_bytes(mmap.as_ref(), owned)?,
                ));
                reader.count()
            },
            ScanSourceRef::Buffer(buffer) => {
                polars_ensure!(!run_async, nyi = "BytesIO with force_async");

                let owned = &mut vec![];
                let reader = polars_io::ndjson::core::JsonLineReader::new(std::io::Cursor::new(
                    maybe_decompress_bytes(buffer, owned)?,
                ));
                reader.count()
            },
        })
        .sum()
}
