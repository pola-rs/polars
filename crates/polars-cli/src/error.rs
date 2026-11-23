use polars::error::{ErrorAttachment, SqlLocationKind};
use polars::prelude::PolarsError;

use crate::script::{Script, Statement};

/// A position in the script, both counted from 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Location {
    pub line: usize,
    pub column: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocationKind {
    Exact,
    /// The clause the cause is in, such as `JOIN`.
    Clause(String),
    /// Only the statement is known; the span is its start.
    Statement,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceLocation {
    pub kind: LocationKind,
    pub start: Location,
    /// After the last character of the span.
    pub end: Location,
    /// The script's line `start.line`.
    pub line: String,
}

/// A failed statement, as data for the printer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ErrorRecord {
    /// The kind of error, for example `SQLSyntaxError`.
    pub class: &'static str,
    pub message: String,
    pub location: Option<SourceLocation>,
    pub expression: Option<String>,
    pub hints: Vec<String>,
    /// For `--verbose`: the plan, the expressions and the context stack.
    pub details: String,
    /// The position of the statement in the script, counted from 1.
    pub statement: usize,
}

impl ErrorRecord {
    pub(crate) fn from_polars(
        error: &PolarsError,
        number: usize,
        statement: &Statement,
        script: &Script,
    ) -> Self {
        let root = root(error);
        let mut message = message_of(root);
        let statement_start = script.location_of(statement.offset);
        let mut location = error.sql_location().and_then(|location| {
            let kind = match &location.kind {
                SqlLocationKind::Exact => LocationKind::Exact,
                SqlLocationKind::Clause(clause) => LocationKind::Clause(clause.clone()),
                _ => LocationKind::Statement,
            };
            let start = in_script(statement_start, location.start.line, location.start.column)?;
            let end = in_script(statement_start, location.end.line, location.end.column)?;
            Some(source_location(kind, start, end, script))
        });
        // sqlparser writes the position into its message, counted from the statement.
        if location.is_none()
            && let Some((rest, line, column)) = parser_position(&message)
            && let Some(start) = in_script(statement_start, line, column)
        {
            message = rest.to_string();
            let token = token_width(script.line(start.line).unwrap_or_default(), start.column);
            let end = Location {
                column: start.column + token,
                ..start
            };
            location = Some(source_location(LocationKind::Exact, start, end, script));
        }

        let text = error
            .clone()
            .retain_attachments(|a| {
                !matches!(
                    a,
                    ErrorAttachment::SqlLocation(_) | ErrorAttachment::Hint(_)
                )
            })
            .to_string();
        // A Python error's text is rebuilt: keep all of it.
        let details = text
            .strip_prefix(&root.to_string())
            .unwrap_or(&text)
            .trim_start_matches('\n')
            .to_string();

        Self {
            class: class_of(error),
            message,
            location,
            expression: innermost_expression(error),
            hints: error.hints().into_iter().map(String::from).collect(),
            details,
            statement: number,
        }
    }

    /// A statement that cannot be split from the script, at `at`.
    pub(crate) fn split(message: String, at: Location, number: usize, script: &Script) -> Self {
        let end = Location {
            column: at.column + 1,
            ..at
        };
        Self {
            location: Some(source_location(LocationKind::Exact, at, end, script)),
            ..Self::bare("SQLInterfaceError", message, number)
        }
    }

    pub(crate) fn internal(panic_message: &str, statement: usize) -> Self {
        let message = format!("{panic_message} (this is a bug in Polars, please report it)");
        Self::bare("InternalError", message, statement)
    }

    /// A statement that was stopped because its spilled data did not fit on disk.
    pub(crate) fn out_of_disk(statement: usize) -> Self {
        let message = "query aborted: spilled data exceeded the out-of-core disk budget, \
                       increase POLARS_OOC_DISK_BUDGET_MB";
        Self::bare("ComputeError", message.to_string(), statement)
    }

    fn bare(class: &'static str, message: String, statement: usize) -> Self {
        Self {
            class,
            message,
            location: None,
            expression: None,
            hints: Vec::new(),
            details: String::new(),
            statement,
        }
    }
}

fn source_location(
    kind: LocationKind,
    start: Location,
    end: Location,
    script: &Script,
) -> SourceLocation {
    SourceLocation {
        kind,
        start,
        end,
        line: script.line(start.line).unwrap_or_default().to_string(),
    }
}

/// The position in the script of `line` and `column` of a statement that starts at `start`.
fn in_script(start: Location, line: u64, column: u64) -> Option<Location> {
    let (line, column) = (usize::try_from(line).ok()?, usize::try_from(column).ok()?);
    Some(match line {
        0 => return None,
        1 => Location {
            line: start.line,
            column: start.column + column.checked_sub(1)?,
        },
        _ => Location {
            line: start.line + line - 1,
            column,
        },
    })
}

/// A sqlparser message without the position it ends with (`... at Line: 2, Column: 10`), and
/// that position.
fn parser_position(message: &str) -> Option<(&str, u64, u64)> {
    // Also inside the message of an error that wraps it.
    if !message.contains("sql parser error: ") {
        return None;
    }
    let (rest, position) = message.rsplit_once(" at Line: ")?;
    let (line, column) = position.split_once(", Column: ")?;
    Some((rest, line.parse().ok()?, column.parse().ok()?))
}

/// The number of characters of the token at `column` of `line`: a word, or one character.
fn token_width(line: &str, column: usize) -> usize {
    let is_word = |c: &char| c.is_alphanumeric() || *c == '_';
    let mut chars = line.chars().skip(column.saturating_sub(1)).peekable();
    match chars.peek() {
        Some(c) if is_word(c) => chars.take_while(is_word).count(),
        _ => 1,
    }
}

fn root(error: &PolarsError) -> &PolarsError {
    match error {
        PolarsError::Context { error, .. }
        | PolarsError::ExprContext { error, .. }
        | PolarsError::Attached { error, .. } => root(error),
        error => error,
    }
}

/// The message of `root`, without the words of its class that its `Display` adds.
fn message_of(root: &PolarsError) -> String {
    use PolarsError::*;
    match root {
        AssertionError(msg)
        | ColumnNotFound(msg)
        | ComputeError(msg)
        | Duplicate(msg)
        | InvalidOperation(msg)
        | NoData(msg)
        | OutOfBounds(msg)
        | SchemaFieldNotFound(msg)
        | SchemaMismatch(msg)
        | ShapeMismatch(msg)
        | SQLInterface(msg)
        | SQLSyntax(msg)
        | StringCacheMismatch(msg)
        | StructFieldNotFound(msg) => msg.to_string(),
        IO { msg: Some(msg), .. } => msg.to_string(),
        error => error.to_string(),
    }
}

fn innermost_expression(error: &PolarsError) -> Option<String> {
    let mut expression = None;
    let mut error = error;
    loop {
        match error {
            PolarsError::ExprContext { error: e, expr } => {
                expression = Some(expr.to_string());
                error = e;
            },
            PolarsError::Context { error: e, .. } | PolarsError::Attached { error: e, .. } => {
                error = e
            },
            _ => return expression,
        }
    }
}

/// The class name of an error, named as the exceptions of Python Polars are.
fn class_of(error: &PolarsError) -> &'static str {
    use PolarsError::*;
    match error {
        AssertionError(_) => "AssertionError",
        ColumnNotFound(_) => "ColumnNotFoundError",
        Duplicate(_) => "DuplicateError",
        InvalidOperation(_) => "InvalidOperationError",
        IO { .. } => "IOError",
        NoData(_) => "NoDataError",
        OutOfBounds(_) => "OutOfBoundsError",
        SchemaFieldNotFound(_) => "SchemaFieldNotFoundError",
        SchemaMismatch(_) => "SchemaError",
        ShapeMismatch(_) => "ShapeError",
        SQLInterface(_) => "SQLInterfaceError",
        SQLSyntax(_) => "SQLSyntaxError",
        StringCacheMismatch(_) => "StringCacheMismatchError",
        StructFieldNotFound(_) => "StructFieldNotFoundError",
        Context { error, .. } | ExprContext { error, .. } | Attached { error, .. } => {
            class_of(error)
        },
        // `ComputeError`, and the variants that exist only with other feature sets.
        _ => "ComputeError",
    }
}

#[cfg(test)]
mod tests {
    use polars::error::ErrString;
    use polars::prelude::polars_err;

    use super::*;

    #[test]
    fn class_of_wrapped_error() {
        let error = polars_err!(ColumnNotFound: "x").context(ErrString::from("while planning"));
        assert_eq!(class_of(&error), "ColumnNotFoundError");
    }
}
