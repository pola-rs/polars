#[cfg(feature = "ipc")]
use arrow::io::ipc::read::get_row_count as count_rows_ipc_sync;
#[cfg(feature = "ipc")]
use polars_core::error::to_compute_err;
#[cfg(feature = "parquet")]
use polars_io::cloud::CloudOptions;
#[cfg(feature = "csv")]
use polars_io::csv::read::count_rows as count_rows_csv;
#[cfg(all(feature = "parquet", feature = "cloud"))]
use polars_io::parquet::read::ParquetAsyncReader;
#[cfg(feature = "parquet")]
use polars_io::parquet::read::ParquetReader;
#[cfg(all(feature = "parquet", feature = "async"))]
use polars_io::pl_async::{get_runtime, with_concurrency_budget};
#[cfg(any(feature = "parquet", feature = "ipc"))]
use polars_io::{utils::is_cloud_url, SerReader};

use super::*;

#[allow(unused_variables)]
pub fn count_rows(paths: &Arc<[PathBuf]>, scan_type: &FileScan) -> PolarsResult<DataFrame> {
    match scan_type {
        #[cfg(feature = "csv")]
        FileScan::Csv {
            options,
            cloud_options,
        } => {
            let parse_options = options.get_parse_options();
            let n_rows: PolarsResult<usize> = paths
                .iter()
                .map(|path| {
                    count_rows_csv(
                        path,
                        parse_options.separator,
                        parse_options.quote_char,
                        parse_options.comment_prefix.as_ref(),
                        parse_options.eol_char,
                        options.has_header,
                    )
                })
                .sum();
            Ok(DataFrame::new(vec![Series::new(
                crate::constants::LEN,
                [n_rows? as IdxSize],
            )])
            .unwrap())
        },
        #[cfg(feature = "parquet")]
        FileScan::Parquet { cloud_options, .. } => {
            let n_rows = count_rows_parquet(paths, cloud_options.as_ref())?;
            Ok(DataFrame::new(vec![Series::new(
                crate::constants::LEN,
                [n_rows as IdxSize],
            )])
            .unwrap())
        },
        #[cfg(feature = "ipc")]
        FileScan::Ipc {
            options,
            cloud_options,
            metadata,
        } => {
            let count: IdxSize = count_rows_ipc(
                paths,
                #[cfg(feature = "cloud")]
                cloud_options.as_ref(),
                metadata.as_ref(),
            )?
            .try_into()
            .map_err(to_compute_err)?;
            Ok(DataFrame::new(vec![Series::new(crate::constants::LEN, [count])]).unwrap())
        },
        FileScan::Anonymous { .. } => {
            unreachable!();
        },
    }
}
#[cfg(feature = "parquet")]
pub(super) fn count_rows_parquet(
    paths: &Arc<[PathBuf]>,
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<usize> {
    if paths.is_empty() {
        return Ok(0);
    };
    let is_cloud = is_cloud_url(paths.first().unwrap().as_path());

    if is_cloud {
        #[cfg(not(feature = "cloud"))]
        panic!("One or more of the cloud storage features ('aws', 'gcp', ...) must be enabled.");

        #[cfg(feature = "cloud")]
        {
            get_runtime().block_on(count_rows_cloud_parquet(paths, cloud_options))
        }
    } else {
        paths
            .iter()
            .map(|path| {
                let file = polars_utils::open_file(path)?;
                let mut reader = ParquetReader::new(file);
                reader.num_rows()
            })
            .sum::<PolarsResult<usize>>()
    }
}

#[cfg(all(feature = "parquet", feature = "async"))]
async fn count_rows_cloud_parquet(
    paths: &Arc<[PathBuf]>,
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<usize> {
    let collection = paths.iter().map(|path| {
        with_concurrency_budget(1, || async {
            let mut reader =
                ParquetAsyncReader::from_uri(&path.to_string_lossy(), cloud_options, None, None)
                    .await?;
            reader.num_rows().await
        })
    });
    futures::future::try_join_all(collection)
        .await
        .map(|rows| rows.iter().sum())
}

#[cfg(feature = "ipc")]
pub(super) fn count_rows_ipc(
    paths: &Arc<[PathBuf]>,
    #[cfg(feature = "cloud")] cloud_options: Option<&CloudOptions>,
    metadata: Option<&arrow::io::ipc::read::FileMetadata>,
) -> PolarsResult<i64> {
    if paths.is_empty() {
        return Ok(0);
    };
    let is_cloud = is_cloud_url(paths.first().unwrap().as_path());

    if is_cloud {
        #[cfg(not(feature = "cloud"))]
        panic!("One or more of the cloud storage features ('aws', 'gcp', ...) must be enabled.");

        #[cfg(feature = "cloud")]
        {
            get_runtime().block_on(count_rows_cloud_ipc(paths, cloud_options, metadata))
        }
    } else {
        paths
            .iter()
            .map(|path| {
                let mut reader = polars_utils::open_file(path)?;
                count_rows_ipc_sync(&mut reader)
            })
            .sum()
    }
}

#[cfg(all(feature = "ipc", feature = "async"))]
async fn count_rows_cloud_ipc(
    paths: &Arc<[PathBuf]>,
    cloud_options: Option<&CloudOptions>,
    metadata: Option<&arrow::io::ipc::read::FileMetadata>,
) -> PolarsResult<i64> {
    use polars_io::ipc::IpcReaderAsync;

    let collection = paths.iter().map(|path| {
        with_concurrency_budget(1, || async {
            let reader = IpcReaderAsync::from_uri(&path.to_string_lossy(), cloud_options).await?;
            reader.count_rows(metadata).await
        })
    });
    futures::future::try_join_all(collection)
        .await
        .map(|rows| rows.iter().sum())
}
