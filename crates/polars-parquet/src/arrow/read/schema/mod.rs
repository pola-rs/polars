//! APIs to handle Parquet <-> Arrow schemas.
use std::sync::Arc;

use arrow::datatypes::{ArrowSchema, TimeUnit};

mod convert;
mod metadata;

pub(crate) use convert::*;
pub use convert::{parquet_to_arrow_schema, parquet_to_arrow_schema_with_options};
pub use metadata::{read_custom_key_value_metadata, read_schema_from_metadata};
use polars_error::{PolarsResult, polars_ensure};
use polars_utils::aliases::{InitHashMaps, PlHashMap, PlHashSet};
use polars_utils::pl_str::PlSmallStr;

use self::metadata::parse_key_value_metadata;
pub use crate::parquet::metadata::{FileMetadata, KeyValue, SchemaDescriptor};
pub use crate::parquet::schema::types::ParquetType;

/// Options when inferring schemas from Parquet.
///
/// The structure mirrors the schema's nesting: `nested` holds the options for named child
/// fields, and the conversion descends through it with [`enter_nested`][Self::enter_nested].
pub struct SchemaInferenceOptions {
    /// The TimeUnit INT96 timestamps convert to (they store a Julian day plus nanoseconds-of-day
    /// and carry no unit of their own).
    ///
    /// This defaults to `TimeUnit::Nanosecond`, but INT96 timestamps outside of the range of years 1678-2262,
    /// will overflow when parsed as `Timestamp(TimeUnit::Nanosecond)`. Setting this to a lower resolution
    /// (e.g. TimeUnit::Milliseconds) will result in loss of precision, but support a larger range of dates
    /// without overflowing when parsing the data.
    pub int96_coerce_to_timeunit: TimeUnit,

    /// Options for the named fields inside this one. Values are fully resolved at construction.
    pub nested: PlHashMap<PlSmallStr, SchemaInferenceOptions>,

    /// The root default options, applying to any field `nested` has no entry for (a field the
    /// scan's schema says nothing about). `None` means these options are the default themselves.
    pub default: Option<Arc<SchemaInferenceOptions>>,
}

impl SchemaInferenceOptions {
    /// The options for the named field `name` inside the current one.
    pub fn enter_nested(&self, name: &str) -> &SchemaInferenceOptions {
        self.nested
            .get(name)
            .or(self.default.as_deref())
            .unwrap_or(self)
    }
}

impl Default for SchemaInferenceOptions {
    fn default() -> Self {
        SchemaInferenceOptions {
            int96_coerce_to_timeunit: TimeUnit::Nanosecond,
            nested: PlHashMap::default(),
            default: None,
        }
    }
}

/// Infers a [`ArrowSchema`] from parquet's [`FileMetadata`].
///
/// This first looks for the metadata key `"ARROW:schema"`; if it does not exist, it converts the
/// Parquet types declared in the file's Parquet schema to Arrow's equivalent.
///
/// # Error
/// - Errors if the key `"ARROW:schema"` exists but is not correctly encoded.
/// - Errors if the parquet schema contains duplicate top-level column names, since
///   the resulting [`ArrowSchema`] cannot represent them.
pub fn infer_schema(file_metadata: &FileMetadata) -> PolarsResult<ArrowSchema> {
    infer_schema_with_options(file_metadata, &SchemaInferenceOptions::default())
}

/// Like [`infer_schema`] but with configurable options which affects the behavior of inference
pub fn infer_schema_with_options(
    file_metadata: &FileMetadata,
    options: &SchemaInferenceOptions,
) -> PolarsResult<ArrowSchema> {
    let fields = file_metadata.schema().fields();
    let mut seen = PlHashSet::with_capacity(fields.len());
    for f in fields {
        polars_ensure!(seen.insert(f.name()), duplicate = f.name());
    }

    let mut metadata = parse_key_value_metadata(file_metadata.key_value_metadata());

    let schema = read_schema_from_metadata(&mut metadata)?;
    match schema {
        Some(schema) => Ok(schema),
        None => parquet_to_arrow_schema_with_options(fields, options),
    }
}
