use polars_io::cloud::CloudOptions;
#[cfg(feature = "csv")]
use polars_io::csv::parser::SplitLines;
use polars_io::parquet::{ParquetAsyncReader, ParquetReader};
use polars_io::pl_async::get_runtime;
#[cfg(feature = "csv")]
use polars_io::utils::get_reader_bytes;
use polars_io::{is_cloud_url, SerReader};

use super::*;

pub fn count_rows(paths: &Arc<[PathBuf]>, scan_type: &FileScan) -> PolarsResult<DataFrame> {
    match scan_type {
        FileScan::Csv { options } => count_rows_csv(paths, options),
        FileScan::Parquet { cloud_options, .. } => {
            count_rows_parquet(paths, cloud_options.as_ref())
        },
        FileScan::Ipc { .. } => {
            unreachable!()
        },
        FileScan::Anonymous { .. } => {
            unreachable!()
        },
    }
}

pub(super) fn count_rows_csv(
    paths: &Arc<[PathBuf]>,
    options: &CsvParserOptions,
) -> PolarsResult<DataFrame> {
    let path = unsafe { paths.get_unchecked(0) };

    let mut reader = polars_utils::open_file(&path)?;
    let reader_bytes = get_reader_bytes(&mut reader)?;

    let row_iterator = SplitLines::new(
        &reader_bytes,
        options.quote_char.unwrap_or(b'"'),
        options.eol_char,
    );
    let n_rows = row_iterator.count() - (options.has_header as usize);
    Ok(DataFrame::new(vec![Series::from_vec("len", vec![n_rows as u32])]).unwrap())
}

pub(super) fn count_rows_parquet(
    paths: &Arc<[PathBuf]>,
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<DataFrame> {
    paths
        .iter()
        .map(|path: &PathBuf| {
            if is_cloud_url(&path) {
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
        .map(|n_rows| DataFrame::new(vec![Series::from_vec("len", vec![n_rows as u32])]).unwrap())
}
