use arrow::datatypes::{ArrowDataType, ArrowSchema, Field, Metadata};
use arrow::io::ipc::read::deserialize_schema;
use base64::engine::general_purpose;
use base64::Engine as _;
use polars_error::{polars_bail, PolarsResult};
use polars_utils::pl_str::PlSmallStr;

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

fn convert_field(field: &mut Field) {
    field.dtype = convert_dtype(std::mem::take(&mut field.dtype));
}

fn convert_dtype(mut dtype: ArrowDataType) -> ArrowDataType {
    use ArrowDataType::*;
    match dtype {
        List(mut field) => {
            convert_field(field.as_mut());
            dtype = LargeList(field);
        },
        LargeList(ref mut field) | FixedSizeList(ref mut field, _) => convert_field(field.as_mut()),
        Struct(ref mut fields) => {
            for field in fields {
                convert_field(field);
            }
        },
        Float16 => dtype = Float32,
        Binary | LargeBinary => dtype = BinaryView,
        Utf8 | LargeUtf8 => dtype = Utf8View,
        Dictionary(_, ref mut dtype, _) | Extension(_, ref mut dtype, _) => {
            let dtype = dtype.as_mut();
            *dtype = convert_dtype(std::mem::take(dtype));
        },
        Map(mut field, _ordered) => {
            // Polars doesn't support Map.
            // A map is physically a `List<Struct<K, V>>`
            // So we read as list.
            convert_field(field.as_mut());
            dtype = LargeList(field);
        },
        _ => {},
    }

    dtype
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
            for field in schema.iter_values_mut() {
                field.dtype = convert_dtype(std::mem::take(&mut field.dtype))
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
                    kv.value.as_ref().map(|value| {
                        (
                            PlSmallStr::from_str(kv.key.as_str()),
                            PlSmallStr::from_str(value.as_str()),
                        )
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}
