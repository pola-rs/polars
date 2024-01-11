use arrow::datatypes::{ArrowDataType, ArrowSchema, Field, Metadata};
use arrow::io::ipc::read::deserialize_schema;
use base64::engine::general_purpose;
use base64::Engine as _;
use polars_error::{polars_bail, PolarsResult};

use super::super::super::ARROW_SCHEMA_META_KEY;
pub use crate::parquet::metadata::KeyValue;

/// Reads an arrow schema from Parquet's file metadata. Returns `None` if no schema was found.
/// # Errors
/// Errors iff the schema cannot be correctly parsed.
pub fn read_schema_from_metadata(metadata: &mut Metadata) -> PolarsResult<Option<ArrowSchema>> {
    metadata
        .remove(ARROW_SCHEMA_META_KEY)
        .map(|encoded| get_arrow_schema_from_metadata(&encoded))
        .transpose()
}

fn convert_field(field: Field) -> Field {
    Field {
        name: field.name,
        data_type: convert_data_type(field.data_type),
        is_nullable: field.is_nullable,
        metadata: field.metadata,
    }
}

fn convert_data_type(data_type: ArrowDataType) -> ArrowDataType {
    use ArrowDataType::*;
    match data_type {
        List(field) => LargeList(Box::new(convert_field(*field))),
        LargeList(field) => LargeList(Box::new(convert_field(*field))),
        Struct(mut fields) => {
            for field in &mut fields {
                *field = convert_field(std::mem::take(field))
            }
            Struct(fields)
        },
        Binary => LargeBinary,
        Utf8 => LargeUtf8,
        Extension(name, data_type, metadata) => {
            let data_type = convert_data_type(*data_type);
            Extension(name, Box::new(data_type), metadata)
        },
        Map(field, _ordered) => {
            // Polars doesn't support Map.
            // A map is physically a `List<Struct<K, V>>`
            // So we read as list.
            LargeList(field)
        },
        dt => dt,
    }
}

/// Try to convert Arrow schema metadata into a schema
fn get_arrow_schema_from_metadata(encoded_meta: &str) -> PolarsResult<ArrowSchema> {
    let decoded = general_purpose::STANDARD.decode(encoded_meta);
    match decoded {
        Ok(bytes) => {
            let slice = if bytes[0..4] == [255u8; 4] {
                &bytes[8..]
            } else {
                bytes.as_slice()
            };
            let mut schema = deserialize_schema(slice).map(|x| x.0)?;
            // Convert the data types to the data types we support.
            for field in schema.fields.iter_mut() {
                field.data_type = convert_data_type(std::mem::take(&mut field.data_type))
            }
            Ok(schema)
        },
        Err(err) => {
            // The C++ implementation returns an error if the schema can't be parsed.
            polars_bail!(InvalidOperation:
                "unable to decode the encoded schema stored in {ARROW_SCHEMA_META_KEY}, {err:?}"
            )
        },
    }
}

pub(super) fn parse_key_value_metadata(key_value_metadata: &Option<Vec<KeyValue>>) -> Metadata {
    key_value_metadata
        .as_ref()
        .map(|key_values| {
            key_values
                .iter()
                .filter_map(|kv| {
                    kv.value
                        .as_ref()
                        .map(|value| (kv.key.clone(), value.clone()))
                })
                .collect()
        })
        .unwrap_or_default()
}
