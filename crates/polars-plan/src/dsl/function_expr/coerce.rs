use polars_core::prelude::*;

pub fn as_struct(s: &[Column]) -> PolarsResult<Column> {
    Ok(StructChunked::from_columns(s[0].name().clone(), s)?.into_column())
}
