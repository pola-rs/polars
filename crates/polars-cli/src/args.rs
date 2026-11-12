use std::ffi::OsString;
use std::fmt;

pub const HELP: &str = "\
Run SQL with Polars.

This command is unstable: its options and output can change in any release.

Usage: polars -c <SQL>

Options:
  -c <SQL>       Run the SQL statement and print its result
  -h, --help     Print this help
  -V, --version  Print the version
";

/// What the command line asks for.
#[derive(Debug, PartialEq, Eq)]
pub enum Invocation {
    Help,
    Version,
    /// Run the SQL given with `-c`.
    Command(String),
}

/// A command line that cannot be run.
#[derive(Debug, PartialEq, Eq)]
pub struct UsageError(String);

impl fmt::Display for UsageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UsageError {}

/// Parses the arguments that follow the program name.
pub fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<Invocation, UsageError> {
    let mut args = args.into_iter();
    let mut command = None;

    while let Some(arg) = args.next() {
        let arg = arg
            .into_string()
            .map_err(|_| UsageError("arguments must be valid UTF-8".to_string()))?;
        match arg.as_str() {
            "-h" | "--help" => return Ok(Invocation::Help),
            "-V" | "--version" => return Ok(Invocation::Version),
            "-c" => {
                let sql = args
                    .next()
                    .ok_or_else(|| UsageError("option '-c' needs a SQL statement".to_string()))?
                    .into_string()
                    .map_err(|_| UsageError("the SQL statement must be valid UTF-8".to_string()))?;
                if command.replace(sql).is_some() {
                    return Err(UsageError(
                        "option '-c' was given more than once".to_string(),
                    ));
                }
            },
            _ if arg.starts_with('-') => {
                return Err(UsageError(format!("unknown option '{arg}'")));
            },
            _ => return Err(UsageError(format!("unexpected argument '{arg}'"))),
        }
    }

    command
        .map(Invocation::Command)
        .ok_or_else(|| UsageError("no SQL was given".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Invocation, UsageError> {
        parse_args(args.iter().map(OsString::from))
    }

    #[test]
    fn command() {
        assert_eq!(
            parse(&["-c", "SELECT 1"]),
            Ok(Invocation::Command("SELECT 1".to_string()))
        );
        // The value of `-c` is taken as is, also when it looks like an option.
        assert_eq!(
            parse(&["-c", "--help"]),
            Ok(Invocation::Command("--help".to_string()))
        );
    }

    #[test]
    fn help_and_version() {
        for flag in ["-h", "--help"] {
            assert_eq!(parse(&[flag]), Ok(Invocation::Help));
        }
        for flag in ["-V", "--version"] {
            assert_eq!(parse(&[flag]), Ok(Invocation::Version));
        }
        assert_eq!(parse(&["-c", "SELECT 1", "--help"]), Ok(Invocation::Help));
    }

    #[test]
    fn usage_errors() {
        let message = |args: &[&str]| parse(args).unwrap_err().to_string();

        assert_eq!(message(&[]), "no SQL was given");
        assert_eq!(message(&["-c"]), "option '-c' needs a SQL statement");
        assert_eq!(
            message(&["-c", "SELECT 1", "-c", "SELECT 2"]),
            "option '-c' was given more than once"
        );
        assert_eq!(message(&["--unknown"]), "unknown option '--unknown'");
        assert_eq!(message(&["query.sql"]), "unexpected argument 'query.sql'");
    }

    #[cfg(unix)]
    #[test]
    fn invalid_utf8() {
        use std::os::unix::ffi::OsStringExt;

        let invalid = || OsString::from_vec(vec![0xff]);
        assert_eq!(
            parse_args([invalid()]).unwrap_err().to_string(),
            "arguments must be valid UTF-8"
        );
        assert_eq!(
            parse_args([OsString::from("-c"), invalid()])
                .unwrap_err()
                .to_string(),
            "the SQL statement must be valid UTF-8"
        );
    }
}
