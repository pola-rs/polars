use std::num::NonZeroUsize;

use arrow::array::StructArray;
use polars_core::prelude::*;

pub(crate) mod buffer;
pub mod core;

pub fn infer_schema<R: std::io::BufRead>(
    reader: &mut R,
    infer_schema_len: Option<NonZeroUsize>,
) -> PolarsResult<Schema> {
    let data_types = polars_json::ndjson::iter_unique_dtypes(reader, infer_schema_len)?;
    let data_type =
        crate::json::infer::data_types_to_supertype(data_types.map(|dt| DataType::from(&dt)))?;
    let schema = StructArray::get_fields(&data_type.to_arrow(CompatLevel::newest()))
        .iter()
        .collect();
    Ok(schema)
}
