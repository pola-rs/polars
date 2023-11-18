use polars_core::prelude::*;

pub fn as_struct(s: &[Series]) -> PolarsResult<Series> {
    Ok(StructChunked::new(s[0].name(), s)?.into_series())
}
