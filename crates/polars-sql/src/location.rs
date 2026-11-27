use polars_error::{ErrorAttachment, PolarsError, SqlLocation, SqlLocationKind, SqlPosition};
use sqlparser::tokenizer::{Location, Span};

/// Attaches `span` unless a more precise location is attached.
pub(crate) fn locate(error: PolarsError, kind: SqlLocationKind, span: Span) -> PolarsError {
    if error.sql_location().is_some() || span.start.line == 0 {
        return error;
    }
    error.with_attachment(ErrorAttachment::SqlLocation(SqlLocation {
        kind,
        start: position(span.start),
        end: position(span.end),
        line: String::new(),
    }))
}

/// Locates `error` at the statement, unless it is located already: for an error about a part
/// of an expression that has no position of its own.
pub(crate) fn at_statement(error: PolarsError) -> PolarsError {
    if error.sql_location().is_some() {
        return error;
    }
    let start = SqlPosition { line: 0, column: 0 };
    error.with_attachment(ErrorAttachment::SqlLocation(SqlLocation {
        kind: SqlLocationKind::Statement,
        start,
        end: start,
        line: String::new(),
    }))
}

/// `complete_location`, or the statement when nothing more precise is known.
pub(crate) fn locate_in_statement(error: PolarsError, sql: &str) -> PolarsError {
    match error.sql_location() {
        Some(_) => complete_location(error, sql),
        None => error.with_attachment(ErrorAttachment::SqlLocation(SqlLocation::statement(sql))),
    }
}

/// Fills in the text of the located line from the statement `sql`.
pub(crate) fn complete_location(error: PolarsError, sql: &str) -> PolarsError {
    let location = match error.sql_location() {
        Some(location)
            if location.line.is_empty() && location.kind == SqlLocationKind::Statement =>
        {
            SqlLocation::statement(sql)
        },
        Some(location) if location.line.is_empty() => {
            let line = (location.start.line as usize).saturating_sub(1);
            SqlLocation {
                line: sql.lines().nth(line).unwrap_or_default().to_string(),
                ..location.clone()
            }
        },
        _ => return error,
    };
    without_positions(error).with_attachment(ErrorAttachment::SqlLocation(location))
}

/// `error` without the positions of macro bodies, if `expanded` put any into the statement:
/// they are in the text that created the macro.
pub(crate) fn without_macro_positions(error: PolarsError, expanded: bool) -> PolarsError {
    match expanded {
        true => without_positions(error),
        false => error,
    }
}

/// Removes the SQL locations of `error` that have no line text yet: positions in a text that
/// is not known here.
pub(crate) fn without_positions(error: PolarsError) -> PolarsError {
    error.retain_attachments(|attachment| {
        !matches!(attachment, ErrorAttachment::SqlLocation(location) if location.line.is_empty())
    })
}

fn position(location: Location) -> SqlPosition {
    SqlPosition {
        line: location.line,
        column: location.column,
    }
}
