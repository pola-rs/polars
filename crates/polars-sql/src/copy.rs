//! `COPY ... TO`: the format and options of what is written.

#[cfg(feature = "csv")]
use std::sync::Arc;

use polars_core::prelude::*;
use polars_lazy::prelude::*;
use sqlparser::ast::{CopyLegacyOption, CopyOption};

use crate::table_functions::FileFormat;

/// A `COPY ... TO STDOUT` statement, for the caller to write to its output (see
/// [`SQLContext::copy_to_stdout`](crate::SQLContext::copy_to_stdout)).
#[derive(Clone)]
pub struct CopyToStdout {
    /// The rows that the statement writes.
    pub rows: LazyFrame,
    /// The format to write them in, with its options.
    pub format: FileWriteFormat,
}

/// How `COPY` writes its rows: in the format that `FORMAT` names, else in that of the extension
/// of `path`, the file it writes. Standard output has no path.
#[allow(unreachable_code, unused_variables)]
pub(crate) fn write_format(
    path: Option<&str>,
    options: &[CopyOption],
    legacy_options: &[CopyLegacyOption],
) -> PolarsResult<FileWriteFormat> {
    polars_ensure!(
        legacy_options.is_empty(),
        SQLInterface: "COPY takes its options in parentheses, as in (FORMAT csv, HEADER false)"
    );
    let mut format = None;
    // the options that only the CSV format has, by name
    let mut csv_options = Vec::new();
    for option in options {
        match option {
            CopyOption::Format(name) => format = Some(name.value.as_str()),
            CopyOption::Header(_) => csv_options.push(("HEADER", option)),
            CopyOption::Delimiter(_) => csv_options.push(("DELIMITER", option)),
            CopyOption::Null(_) => csv_options.push(("NULL", option)),
            CopyOption::Quote(_) => csv_options.push(("QUOTE", option)),
            _ => {
                // an option prints as its name and value
                let option = option.to_string();
                let name = option.split([' ', '(']).next().unwrap_or(&option);
                polars_bail!(SQLInterface: "COPY option {} is not supported", name)
            },
        }
    }

    let names = FileFormat::NAMES.join(", ");
    let format = match (format, path) {
        (Some(name), _) => FileFormat::from_name(name).ok_or_else(|| {
            polars_err!(
                SQLInterface: "COPY format '{}' is not supported; the formats are: {}", name, names
            )
        })?,
        (None, Some(path)) => FileFormat::of_path(path).ok_or_else(|| {
            polars_err!(
                SQLInterface:
                "COPY cannot tell the format of '{}' from its extension; add (FORMAT <name>), with one of: {}",
                path, names
            )
        })?,
        (None, None) => polars_bail!(
            SQLInterface: "COPY TO STDOUT needs a format; add (FORMAT <name>), with one of: {}", names
        ),
    };

    #[cfg(feature = "csv")]
    if let FileFormat::Csv = format {
        // `FORMAT csv` holds for a path of any extension
        let mut csv = CsvWriterOptions {
            check_extension: false,
            ..Default::default()
        };
        let serialize = Arc::make_mut(&mut csv.serialize_options);
        for (name, option) in csv_options {
            match option {
                CopyOption::Header(header) => csv.include_header = *header,
                CopyOption::Delimiter(c) => serialize.separator = single_byte(*c, name)?,
                CopyOption::Quote(c) => serialize.quote_char = single_byte(*c, name)?,
                CopyOption::Null(null) => serialize.null = null.as_str().into(),
                _ => unreachable!(),
            }
        }
        return Ok(FileWriteFormat::Csv(csv));
    }
    if let Some((name, _)) = csv_options.first() {
        polars_bail!(SQLInterface: "COPY option {} applies to the CSV format only", name);
    }

    Ok(match format {
        #[cfg(feature = "csv")]
        FileFormat::Csv => unreachable!(),
        #[cfg(feature = "parquet")]
        FileFormat::Parquet => FileWriteFormat::Parquet(Default::default()),
        #[cfg(feature = "ipc")]
        FileFormat::Ipc => FileWriteFormat::Ipc(Default::default()),
        #[cfg(feature = "json")]
        FileFormat::Ndjson => FileWriteFormat::NDJson(Default::default()),
    })
}

/// The byte that the CSV writer takes for the character of option `name`.
#[cfg(feature = "csv")]
fn single_byte(c: char, name: &str) -> PolarsResult<u8> {
    u8::try_from(c)
        .ok()
        .filter(u8::is_ascii)
        .ok_or_else(|| polars_err!(SQLInterface: "COPY option {} takes one ASCII character", name))
}
