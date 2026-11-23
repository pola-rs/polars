//! `COPY`: the format and options of what `COPY ... TO` writes and `COPY ... FROM` reads.

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
    let (format, csv_options) = format_and_csv_options(path, options, legacy_options)?;

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

/// How `COPY <table> FROM` reads the file at `path`: in the format that `FORMAT` names, else in
/// that of its extension. A CSV file is read here, by position into `schema`, the types of the
/// columns it fills. `None` for another format, which is read as `FROM '<file>'` reads it.
#[allow(unreachable_code, unused_variables)]
pub(crate) fn read_rows(
    path: &str,
    schema: &Schema,
    options: &[CopyOption],
    legacy_options: &[CopyLegacyOption],
) -> PolarsResult<(FileFormat, Option<LazyFrame>)> {
    let (format, csv_options) = format_and_csv_options(Some(path), options, legacy_options)?;

    #[cfg(feature = "csv")]
    if let FileFormat::Csv = format {
        use polars_io::csv::read::NullValues;
        use polars_utils::pl_path::PlRefPath;

        use crate::table_functions::expand_home_dir;

        let path = expand_home_dir(PlRefPath::new(path))?;
        // As in Postgres: no header unless the statement says so, fields by position, an
        // unquoted empty field is NULL and a quoted one an empty string.
        let mut header = false;
        // One file, as in Postgres: no glob, so a missing file is an error.
        let mut reader = LazyCsvReader::new_paths(vec![path].into())
            .with_glob(false)
            .with_has_header(false)
            .with_schema(Some(Arc::new(schema.clone())))
            .with_raise_if_empty(false);
        for (name, option) in csv_options {
            match option {
                CopyOption::Header(h) => header = *h,
                CopyOption::Delimiter(c) => reader = reader.with_separator(single_byte(*c, name)?),
                CopyOption::Quote(c) => {
                    reader = reader.with_quote_char(Some(single_byte(*c, name)?))
                },
                // As in Postgres: then only the marker is NULL, and an empty field is an empty
                // string.
                CopyOption::Null(null) if !null.is_empty() => {
                    reader = reader
                        .with_null_values(Some(NullValues::AllColumnsSingle(null.as_str().into())))
                        .with_missing_is_null(false)
                },
                CopyOption::Null(_) => {},
                _ => unreachable!(),
            }
        }
        // The first line is skipped whatever its names, as with Postgres's `HEADER true`.
        let reader = reader.with_skip_rows(usize::from(header));
        return Ok((format, Some(reader.finish()?)));
    }
    if let Some((name, _)) = csv_options.first() {
        polars_bail!(SQLInterface: "COPY option {} applies to the CSV format only", name);
    }
    Ok((format, None))
}

/// The options that `COPY` takes, for errors.
const COPY_OPTIONS: &str = "the options FORMAT, HEADER, DELIMITER, QUOTE and NULL";

/// A parser error's `message`, with the `COPY` options if it stopped at an unknown one.
pub(crate) fn with_copy_options(message: String) -> String {
    const EXPECTED: &str = "sql parser error: Expected: option, found: ";
    if !message.starts_with(EXPECTED) {
        return message;
    }
    let start = EXPECTED.len();
    let (head, position) = match message.rfind(" at Line: ") {
        Some(i) if i > start => message.split_at(i),
        _ => (message.as_str(), ""),
    };
    let found = head[start..].split_whitespace().next().unwrap_or_default();
    let hint = match found.to_ascii_uppercase().as_str() {
        "SEPARATOR" | "SEP" | "DELIM" | "DELIMETER" => " (DELIMITER sets the separator)",
        "QUOTE_CHAR" | "QUOTECHAR" | "QUOTE_STYLE" => " (QUOTE sets the quote character)",
        "NULLSTR" | "NULL_VALUE" | "NULL_VALUES" | "NULLVALUE" => " (NULL sets the text of a NULL)",
        "HAS_HEADER" | "HEADERS" => " (HEADER says whether there is a header row)",
        _ => "",
    };
    format!("{head}; COPY takes {COPY_OPTIONS}{hint}{position}")
}

/// The format of a `COPY`: the one that `FORMAT` names, else that of the extension of `path`
/// (standard output has none). With the options that only the CSV format has, by name.
fn format_and_csv_options<'a>(
    path: Option<&str>,
    options: &'a [CopyOption],
    legacy_options: &[CopyLegacyOption],
) -> PolarsResult<(FileFormat, Vec<(&'static str, &'a CopyOption)>)> {
    polars_ensure!(
        legacy_options.is_empty(),
        SQLInterface: "COPY takes its options in parentheses, as in (FORMAT csv, HEADER false)"
    );
    let mut format = None;
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
                polars_bail!(SQLInterface: "COPY option {} is not supported; COPY takes {}", name, COPY_OPTIONS)
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
    Ok((format, csv_options))
}

/// The byte that the CSV reader or writer takes for the character of option `name`.
#[cfg(feature = "csv")]
fn single_byte(c: char, name: &str) -> PolarsResult<u8> {
    u8::try_from(c)
        .ok()
        .filter(u8::is_ascii)
        .ok_or_else(|| polars_err!(SQLInterface: "COPY option {} takes one ASCII character", name))
}
