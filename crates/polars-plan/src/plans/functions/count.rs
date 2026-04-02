use polars_io::cloud::CloudOptions;

use super::*;

pub fn count_rows(
    sources: &ScanSources,
    scan_type: &FileScanIR,
    alias: Option<PlSmallStr>,
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<DataFrame> {
    feature_gated!("csv", {
        let count: PolarsResult<usize> = match scan_type {
            #[cfg(feature = "csv")]
            FileScanIR::Csv { options } => count_all_rows_csv(sources, options, cloud_options),
            _ => unreachable!(),
        };
        let count = count?;
        let count: IdxSize = count.try_into().map_err(
            |_| polars_err!(ComputeError: "count of {} exceeded maximum row size", count),
        )?;
        let column_name = alias.unwrap_or(PlSmallStr::from_static(crate::constants::LEN));

        Ok(unsafe { DataFrame::new_unchecked(1, vec![Column::new(column_name, [count])]) })
    })
}

#[cfg(feature = "csv")]
fn count_all_rows_csv(
    sources: &ScanSources,
    options: &polars_io::prelude::CsvReadOptions,
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<usize> {
    let run_async =
        sources.is_cloud_url() || (sources.is_paths() && polars_config::config().force_async());

    if run_async {
        if sources.as_paths().is_some() {
            let sources_clone = sources.clone();
            feature_gated!("cloud", {
                polars_io::pl_async::get_runtime().block_in_place_on(
                    polars_io::file_cache::init_entries_from_uri_list(
                        (0..sources_clone.len()).map(move |i| {
                            sources_clone.as_paths().unwrap().get(i).unwrap().clone()
                        }),
                        cloud_options,
                    ),
                )?;
            })
        }
    }

    let parse_options = options.get_parse_options();

    sources
        .iter()
        .map(|source| match source {
            ScanSourceRef::Path(path) => polars_io::csv::read::count_rows(
                path.clone(),
                parse_options.quote_char,
                parse_options.comment_prefix.as_ref(),
                parse_options.eol_char,
                options.has_header,
                options.skip_lines,
                options.skip_rows,
                options.skip_rows_after_header,
                options.raise_if_empty,
            ),
            _ => {
                let memslice = source.to_memslice()?;

                polars_io::csv::read::count_rows_from_slice_par(
                    memslice,
                    parse_options.quote_char,
                    parse_options.comment_prefix.as_ref(),
                    parse_options.eol_char,
                    options.has_header,
                    options.skip_lines,
                    options.skip_rows,
                    options.skip_rows_after_header,
                    options.raise_if_empty,
                )
            },
        })
        .sum()
}
