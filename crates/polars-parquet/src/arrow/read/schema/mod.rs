//! APIs to handle Parquet <-> Arrow schemas.
use arrow::datatypes::{ArrowSchema, TimeUnit};

mod convert;
mod metadata;

pub(crate) use convert::*;
pub use convert::{parquet_to_arrow_schema, parquet_to_arrow_schema_with_options};
pub use metadata::read_schema_from_metadata;
use polars_error::PolarsResult;

use self::metadata::parse_key_value_metadata;
pub use crate::parquet::metadata::{FileMetaData, KeyValue, SchemaDescriptor};
pub use crate::parquet::schema::types::ParquetType;

/// Options when inferring schemas from Parquet
pub struct SchemaInferenceOptions {
    /// When inferring schemas from the Parquet INT96 timestamp type, this is the corresponding TimeUnit
    /// in the inferred Arrow Timestamp type.
    ///
    /// This defaults to `TimeUnit::Nanosecond`, but INT96 timestamps outside of the range of years 1678-2262,
    /// will overflow when parsed as `Timestamp(TimeUnit::Nanosecond)`. Setting this to a lower resolution
    /// (e.g. TimeUnit::Milliseconds) will result in loss of precision, but support a larger range of dates
    /// without overflowing when parsing the data.
    pub int96_coerce_to_timeunit: TimeUnit,
}

impl Default for SchemaInferenceOptions {
    fn default() -> Self {
        SchemaInferenceOptions {
            int96_coerce_to_timeunit: TimeUnit::Nanosecond,
        }
    }
}

/// Infers a [`ArrowSchema`] from parquet's [`FileMetaData`].
///
/// This first looks for the metadata key `"ARROW:schema"`; if it does not exist, it converts the
/// Parquet types declared in the file's Parquet schema to Arrow's equivalent.
///
/// # Error
/// This function errors iff the key `"ARROW:schema"` exists but is not correctly encoded,
/// indicating that that the file's arrow metadata was incorrectly written.
pub fn infer_schema(file_metadata: &FileMetaData) -> PolarsResult<ArrowSchema> {
    infer_schema_with_options(file_metadata, &None)
}

/// Like [`infer_schema`] but with configurable options which affects the behavior of inference
pub fn infer_schema_with_options(
    file_metadata: &FileMetaData,
    options: &Option<SchemaInferenceOptions>,
) -> PolarsResult<ArrowSchema> {
    let mut metadata = parse_key_value_metadata(file_metadata.key_value_metadata());

    let schema = read_schema_from_metadata(&mut metadata)?;
    Ok(schema.unwrap_or_else(|| {
        let fields = parquet_to_arrow_schema_with_options(file_metadata.schema().fields(), options);
        ArrowSchema { fields, metadata }
    }))
}
