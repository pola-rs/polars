use clap::Parser;

/// The command line of `polars`.
#[derive(Debug, Parser)]
#[command(
    name = "polars",
    bin_name = "polars",
    version,
    about = "Run SQL with Polars",
    after_help = "This command is unstable: its options and output can change in any release."
)]
pub struct Args {
    /// Run the SQL statement and print its result
    #[arg(short = 'c', value_name = "SQL")]
    pub command: String,
}

#[cfg(test)]
mod tests {
    use clap::error::ErrorKind;

    use super::*;

    fn parse(args: &[&str]) -> Result<Args, clap::Error> {
        Args::try_parse_from(std::iter::once(&"polars").chain(args))
    }

    #[test]
    fn command() {
        assert_eq!(parse(&["-c", "SELECT 1"]).unwrap().command, "SELECT 1");
        // The value of `-c` is taken as is, also when it starts with dashes.
        assert_eq!(parse(&["-c=--x"]).unwrap().command, "--x");
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
        for args in [&[][..], &["-c"], &["--unknown"], &["query.sql"]] {
            assert_eq!(parse(args).unwrap_err().exit_code(), 2, "{args:?}");
        }
    }
}
