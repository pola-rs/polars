use std::num::NonZeroUsize;

use arrow::array::StructArray;
use polars_core::prelude::*;

pub(crate) mod buffer;
pub mod core;

pub fn infer_schema<R: std::io::BufRead>(
    reader: &mut R,
    infer_schema_len: Option<NonZeroUsize>,
) -> PolarsResult<Schema> {
    let dtypes = polars_json::ndjson::iter_unique_dtypes(reader, infer_schema_len)?;
    let dtype =
        crate::json::infer::dtypes_to_supertype(dtypes.map(|dt| DataType::from_arrow_dtype(&dt)))?;
    let schema = StructArray::get_fields(&dtype.to_arrow(CompatLevel::newest()))
        .iter()
        .map(Into::<Field>::into)
        .collect();
    Ok(schema)
}
