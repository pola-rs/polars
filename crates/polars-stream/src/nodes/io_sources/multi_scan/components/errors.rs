use polars_error::{PolarsError, polars_err};

pub fn missing_column_err(missing_column_name: &str) -> PolarsError {
    polars_err!(
        ColumnNotFound:
        "did not find column {}, consider passing `missing_columns='insert'`",
        missing_column_name,
    )
}

pub fn extra_column_err(extra_column_name: &str) -> PolarsError {
    polars_err!(
        SchemaMismatch:
        "extra column in file outside of expected schema: {}, \
        hint: specify this column in the schema, or pass \
        extra_columns='ignore' in scan options",
        extra_column_name,
    )
}
