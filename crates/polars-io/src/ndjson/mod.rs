use core::json_lines;
use std::num::NonZeroUsize;

use arrow::array::StructArray;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ExternalCompression;

pub(crate) mod buffer;
pub mod core;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct NDJsonWriterOptions {
    pub compression: ExternalCompression,
    #[cfg_attr(feature = "serde", serde(default))]
    pub check_extension: bool,
}

pub fn infer_schema<R: std::io::BufRead>(
    reader: &mut R,
    infer_schema_len: Option<NonZeroUsize>,
) -> PolarsResult<Schema> {
    let dtypes = polars_json::ndjson::iter_unique_dtypes(reader, infer_schema_len)?;
    let dtype =
        crate::json::infer::dtypes_to_supertype(dtypes.map(|dt| DataType::from_arrow_dtype(&dt)))?;

    if !matches!(&dtype, DataType::Struct(_)) {
        polars_bail!(ComputeError: "NDJSON line expected to contain JSON object: {dtype}");
    }

    let schema = StructArray::get_fields(&dtype.to_arrow(CompatLevel::newest()))
        .iter()
        .map(Into::<Field>::into)
        .collect();
    Ok(schema)
}

/// Count the number of rows. The slice passed must represent the entire file.
/// This does not check if the lines are valid NDJSON - it assumes that is the case.
pub fn count_rows(full_bytes: &[u8]) -> usize {
    json_lines(full_bytes).count()
}
