use super::*;

pub fn count_rows(
    sources: &ScanSources,
    scan_type: &FileScanIR,
    alias: Option<PlSmallStr>,
) -> PolarsResult<DataFrame> {
    feature_gated!("csv", {
        let count: PolarsResult<usize> = match scan_type {
            #[cfg(feature = "csv")]
            FileScanIR::Csv { options } => count_all_rows_csv(sources, options),
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
) -> PolarsResult<usize> {
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
                )
            },
        })
        .sum()
}
