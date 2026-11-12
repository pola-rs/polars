use polars::prelude::PolarsError;

/// A position in the SQL text, both counted from 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Location {
    pub line: usize,
    pub column: usize,
}

/// A failed statement, as data for the printer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ErrorRecord {
    /// The kind of error, for example `SQLSyntaxError`.
    pub class: &'static str,
    pub message: String,
    pub hint: Option<String>,
    pub location: Option<Location>,
    /// The position of the statement in the script, counted from 1.
    pub statement: usize,
}

impl ErrorRecord {
    pub fn from_polars(error: &PolarsError, statement: usize) -> Self {
        Self {
            class: class_of(error),
            message: error.to_string(),
            hint: None,
            location: None,
            statement,
        }
    }

    /// A statement that panicked.
    pub(crate) fn internal(panic_message: &str, statement: usize) -> Self {
        Self {
            class: "InternalError",
            message: format!("{panic_message} (this is a bug in Polars, please report it)"),
            hint: None,
            location: None,
            statement,
        }
    }

    /// A statement that was stopped because its spilled data did not fit on disk.
    pub(crate) fn out_of_disk(statement: usize) -> Self {
        Self {
            class: "ComputeError",
            message: "query aborted: spilled data exceeded the out-of-core disk budget, \
                      increase POLARS_OOC_DISK_BUDGET_MB"
                .to_string(),
            hint: None,
            location: None,
            statement,
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
        Context { error, .. } | ExprContext { error, .. } => class_of(error),
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
    fn class_and_message() {
        let record = ErrorRecord::from_polars(&polars_err!(SQLSyntax: "unexpected token"), 3);
        assert_eq!(
            record,
            ErrorRecord {
                class: "SQLSyntaxError",
                message: "unexpected token".to_string(),
                hint: None,
                location: None,
                statement: 3,
            }
        );

        let record = ErrorRecord::from_polars(&polars_err!(ComputeError: "overflow"), 1);
        assert_eq!(record.class, "ComputeError");
    }

    #[test]
    fn class_of_wrapped_error() {
        let error = polars_err!(ColumnNotFound: "x").context(ErrString::from("while planning"));
        assert_eq!(class_of(&error), "ColumnNotFoundError");
    }
}
