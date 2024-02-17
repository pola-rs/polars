#[cfg(feature = "parquet")]
use arrow::io::ipc::read::get_row_count as count_rows_ipc;
#[cfg(feature = "parquet")]
use polars_io::cloud::CloudOptions;
#[cfg(feature = "csv")]
use polars_io::csv::count_rows as count_rows_csv;
#[cfg(feature = "parquet")]
use polars_io::parquet::{ParquetAsyncReader, ParquetReader};
#[cfg(feature = "parquet")]
use polars_io::pl_async::get_runtime;
#[cfg(feature = "parquet")]
use polars_io::{is_cloud_url, SerReader};

use super::*;

#[allow(unused_variables)]
pub fn count_rows(paths: &Arc<[PathBuf]>, scan_type: &FileScan) -> PolarsResult<DataFrame> {
    match scan_type {
        #[cfg(feature = "csv")]
        FileScan::Csv { options } => {
            // SAFETY
            // should be exactly one path when reading csv
            let path = unsafe { paths.get_unchecked(0) };
            let n_rows = count_rows_csv(
                path,
                options.quote_char,
                options.comment_prefix.as_ref(),
                options.eol_char,
                options.has_header,
            )?;
            Ok(DataFrame::new(vec![Series::new("len", [n_rows as IdxSize])]).unwrap())
        },
        #[cfg(feature = "parquet")]
        FileScan::Parquet { cloud_options, .. } => {
            let n_rows = count_rows_parquet(paths, cloud_options.as_ref())?;
            Ok(DataFrame::new(vec![Series::new("len", [n_rows as IdxSize])]).unwrap())
        },
        #[cfg(feature = "ipc")]
        FileScan::Ipc { options } => {
            let n_rows: PolarsResult<i64> = paths
                .iter()
                .map(|path| {
                    let mut reader = polars_utils::open_file(path)?;
                    count_rows_ipc(&mut reader)
                })
                .sum();
            Ok(DataFrame::new(vec![Series::new("len", [n_rows? as IdxSize])]).unwrap())
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
    paths
        .iter()
        .map(|path: &PathBuf| {
            if is_cloud_url(path) {
                #[cfg(not(feature = "cloud"))]
                panic!(
                "One or more of the cloud storage features ('aws', 'gcp', ...) must be enabled."
            );

                #[cfg(feature = "cloud")]
                {
                    let uri = path.to_string_lossy();
                    get_runtime().block_on(async {
                        let mut reader =
                            ParquetAsyncReader::from_uri(&uri, cloud_options, None, None).await?;
                        reader.num_rows().await
                    })
                }
            } else {
                let file = polars_utils::open_file(path)?;
                let mut reader = ParquetReader::new(file);
                reader.num_rows()
            }
        })
        .sum::<PolarsResult<usize>>()
}
