use std::path::PathBuf;

use clap::Parser;

const AFTER_HELP: &str = "\
A script can hold several statements, separated by ';'. They run in order and
share one session. The run stops at the first statement that fails.

Without -c or -f, the script is read from standard input.

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

impl Args {
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
            &["query.sql"],
        ] {
            assert_eq!(parse(args).unwrap_err().exit_code(), 2, "{args:?}");
        }
    }
}
