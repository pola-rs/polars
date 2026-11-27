use std::path::PathBuf;

use clap::Parser;
use clap::builder::FalseyValueParser;

use crate::printer::Format;

const AFTER_HELP: &str = "\
A file is a table: name it by its path in single quotes. Its format comes from
the extension: .parquet, .csv, .ndjson, .jsonl, .ipc, .arrow or .feather. A
path can be a glob or a cloud URL.
  polars -c \"SELECT * FROM 'sales.parquet' LIMIT 5\"
  polars -c \"DESCRIBE 'events.ndjson'\"

To set options, read the file with read_csv, read_parquet, read_ndjson or
read_ipc. An unknown option is an error that lists the valid ones.
  polars -c \"SELECT * FROM read_csv('sales.csv', separator => ';')\"

SHOW FUNCTIONS lists the SQL functions, and with LIKE those whose name matches.
  polars -c \"SHOW FUNCTIONS LIKE '%date%'\"

COPY writes a result to a file, in the format of the file's extension.
  polars -c \"COPY (SELECT * FROM 'sales.parquet') TO 'sales.csv'\"

Quote a name that is a SQL keyword, such as \"user\". A dot reads a field of a
struct, as in \"user\".country.

A script can hold several statements, separated by ';'. They run in order and
share one session. The run stops at the first statement that fails.

Without -c or -f, the script is read from standard input.

COPY ... TO STDOUT (FORMAT <name>) writes its rows to standard output in the
format it names, not in that of --format. With a binary format, such as
parquet, it must be the last statement and the only one that returns rows.

Standard output holds only the rows of queries, in every format. Statements
that change the session or write a file, such as CREATE TABLE and COPY ... TO a
file, print nothing when they succeed.

Exit codes:
  0    every statement succeeded
  1    a statement failed
  2    the command line could not be used
  101  internal error (a bug)
  130  interrupted

This command is unstable: its options and output can change in any release.";

/// The command line of `polars`.
#[derive(Debug, Parser)]
#[command(
    name = "polars",
    bin_name = "polars",
    version,
    about = "Run SQL with Polars",
    after_help = AFTER_HELP
)]
pub struct Args {
    /// Run the SQL given on the command line
    #[arg(short = 'c', long, value_name = "SQL", conflicts_with = "file")]
    command: Option<String>,

    /// Run the SQL script in FILE
    #[arg(short = 'f', long, value_name = "FILE")]
    file: Option<PathBuf>,

    /// How results are written
    #[arg(long, value_enum, value_name = "FMT", default_value_t)]
    format: Format,

    /// Refuse statements that write files, such as COPY to a file
    #[arg(long, env = "POLARS_READ_ONLY", value_parser = FalseyValueParser::new())]
    read_only: bool,

    /// Show the plan and the context of an error
    #[arg(long)]
    verbose: bool,
}

/// Where the SQL script comes from.
#[derive(Debug, PartialEq, Eq)]
pub enum Input {
    /// The value of `-c`.
    Command(String),
    /// The file named with `-f`.
    File(PathBuf),
    Stdin,
}

impl Input {
    pub fn name(&self) -> String {
        match self {
            Input::Command(_) => "<command>".to_string(),
            Input::File(path) => path.display().to_string(),
            Input::Stdin => "<stdin>".to_string(),
        }
    }
}

impl Args {
    pub fn format(&self) -> Format {
        self.format
    }

    pub fn read_only(&self) -> bool {
        self.read_only
    }

    pub fn read_only_origin(&self) -> Option<&'static str> {
        let flag = || {
            std::env::args_os().skip(1).any(|arg| {
                arg == "--read-only" || arg.to_string_lossy().starts_with("--read-only=")
            })
        };
        match (self.read_only, flag()) {
            (false, _) => None,
            (true, true) => Some("--read-only"),
            (true, false) => Some("POLARS_READ_ONLY"),
        }
    }

    pub fn verbose(&self) -> bool {
        self.verbose
    }

    pub fn input(self) -> Input {
        match (self.command, self.file) {
            (Some(sql), _) => Input::Command(sql),
            (None, Some(path)) => Input::File(path),
            (None, None) => Input::Stdin,
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::error::ErrorKind;

    use super::*;

    fn parse(args: &[&str]) -> Result<Args, clap::Error> {
        Args::try_parse_from(std::iter::once(&"polars").chain(args))
    }

    fn input(args: &[&str]) -> Input {
        parse(args).unwrap().input()
    }

    #[test]
    fn inputs() {
        let command = Input::Command("SELECT 1".to_string());
        assert_eq!(input(&["-c", "SELECT 1"]), command);
        assert_eq!(input(&["--command", "SELECT 1"]), command);

        let file = Input::File(PathBuf::from("query.sql"));
        assert_eq!(input(&["-f", "query.sql"]), file);
        assert_eq!(input(&["--file", "query.sql"]), file);

        assert_eq!(input(&[]), Input::Stdin);
    }

    #[test]
    fn formats() {
        assert_eq!(parse(&[]).unwrap().format(), Format::Table);
        for (name, format) in [
            ("table", Format::Table),
            ("csv", Format::Csv),
            ("ndjson", Format::Ndjson),
            ("parquet", Format::Parquet),
            ("arrow-stream", Format::ArrowStream),
        ] {
            assert_eq!(parse(&["--format", name]).unwrap().format(), format);
            assert_eq!(format.to_string(), name);
        }
    }

    #[test]
    fn read_only() {
        assert!(parse(&["--read-only"]).unwrap().read_only());
    }

    #[test]
    fn help_and_version() {
        for (flag, kind) in [
            ("-h", ErrorKind::DisplayHelp),
            ("--help", ErrorKind::DisplayHelp),
            ("-V", ErrorKind::DisplayVersion),
            ("--version", ErrorKind::DisplayVersion),
        ] {
            let request = parse(&[flag]).unwrap_err();
            assert_eq!(request.kind(), kind);
            assert_eq!(request.exit_code(), 0);
        }
    }

    #[test]
    fn usage_errors() {
        for args in [
            &["-c"][..],
            &["-c", "SELECT 1", "-f", "query.sql"],
            &["-c", "SELECT 1", "-c", "SELECT 2"],
            &["--unknown"],
            &["--format", "xml"],
            // Nothing on the command line turns it off again.
            &["--read-only=false"],
            &["query.sql"],
        ] {
            assert_eq!(parse(args).unwrap_err().exit_code(), 2, "{args:?}");
        }
    }
}
