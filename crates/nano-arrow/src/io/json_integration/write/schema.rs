use serde_json::{json, Map, Value};

use crate::datatypes::{DataType, Field, IntervalUnit, Metadata, Schema, TimeUnit};
use crate::io::ipc::IpcField;
use crate::io::json_integration::ArrowJsonSchema;

use super::super::{ArrowJsonField, ArrowJsonFieldDictionary, IntegerType};

fn serialize_data_type(data_type: &DataType) -> Value {
    match data_type {
        DataType::Null => json!({"name": "null"}),
        DataType::Boolean => json!({"name": "bool"}),
        DataType::Int8 => json!({"name": "int", "bitWidth": 8, "isSigned": true}),
        DataType::Int16 => json!({"name": "int", "bitWidth": 16, "isSigned": true}),
        DataType::Int32 => json!({"name": "int", "bitWidth": 32, "isSigned": true}),
        DataType::Int64 => json!({"name": "int", "bitWidth": 64, "isSigned": true}),
        DataType::UInt8 => json!({"name": "int", "bitWidth": 8, "isSigned": false}),
        DataType::UInt16 => json!({"name": "int", "bitWidth": 16, "isSigned": false}),
        DataType::UInt32 => json!({"name": "int", "bitWidth": 32, "isSigned": false}),
        DataType::UInt64 => json!({"name": "int", "bitWidth": 64, "isSigned": false}),
        DataType::Float16 => json!({"name": "floatingpoint", "precision": "HALF"}),
        DataType::Float32 => json!({"name": "floatingpoint", "precision": "SINGLE"}),
        DataType::Float64 => json!({"name": "floatingpoint", "precision": "DOUBLE"}),
        DataType::Utf8 => json!({"name": "utf8"}),
        DataType::LargeUtf8 => json!({"name": "largeutf8"}),
        DataType::Binary => json!({"name": "binary"}),
        DataType::LargeBinary => json!({"name": "largebinary"}),
        DataType::FixedSizeBinary(byte_width) => {
            json!({"name": "fixedsizebinary", "byteWidth": byte_width})
        }
        DataType::Struct(_) => json!({"name": "struct"}),
        DataType::Union(_, _, _) => json!({"name": "union"}),
        DataType::Map(_, _) => json!({"name": "map"}),
        DataType::List(_) => json!({ "name": "list"}),
        DataType::LargeList(_) => json!({ "name": "largelist"}),
        DataType::FixedSizeList(_, length) => {
            json!({"name":"fixedsizelist", "listSize": length})
        }
        DataType::Time32(unit) => {
            json!({"name": "time", "bitWidth": 32, "unit": match unit {
                TimeUnit::Second => "SECOND",
                TimeUnit::Millisecond => "MILLISECOND",
                TimeUnit::Microsecond => "MICROSECOND",
                TimeUnit::Nanosecond => "NANOSECOND",
            }})
        }
        DataType::Time64(unit) => {
            json!({"name": "time", "bitWidth": 64, "unit": match unit {
                TimeUnit::Second => "SECOND",
                TimeUnit::Millisecond => "MILLISECOND",
                TimeUnit::Microsecond => "MICROSECOND",
                TimeUnit::Nanosecond => "NANOSECOND",
            }})
        }
        DataType::Date32 => {
            json!({"name": "date", "unit": "DAY"})
        }
        DataType::Date64 => {
            json!({"name": "date", "unit": "MILLISECOND"})
        }
        DataType::Timestamp(unit, None) => {
            json!({"name": "timestamp", "unit": match unit {
                TimeUnit::Second => "SECOND",
                TimeUnit::Millisecond => "MILLISECOND",
                TimeUnit::Microsecond => "MICROSECOND",
                TimeUnit::Nanosecond => "NANOSECOND",
            }})
        }
        DataType::Timestamp(unit, Some(tz)) => {
            json!({"name": "timestamp", "unit": match unit {
                TimeUnit::Second => "SECOND",
                TimeUnit::Millisecond => "MILLISECOND",
                TimeUnit::Microsecond => "MICROSECOND",
                TimeUnit::Nanosecond => "NANOSECOND",
            }, "timezone": tz})
        }
        DataType::Interval(unit) => json!({"name": "interval", "unit": match unit {
            IntervalUnit::YearMonth => "YEAR_MONTH",
            IntervalUnit::DayTime => "DAY_TIME",
            IntervalUnit::MonthDayNano => "MONTH_DAY_NANO",
        }}),
        DataType::Duration(unit) => json!({"name": "duration", "unit": match unit {
            TimeUnit::Second => "SECOND",
            TimeUnit::Millisecond => "MILLISECOND",
            TimeUnit::Microsecond => "MICROSECOND",
            TimeUnit::Nanosecond => "NANOSECOND",
        }}),
        DataType::Dictionary(_, _, _) => json!({ "name": "dictionary"}),
        DataType::Decimal(precision, scale) => {
            json!({"name": "decimal", "precision": precision, "scale": scale})
        }
        DataType::Decimal256(precision, scale) => {
            json!({"name": "decimal", "precision": precision, "scale": scale, "bit_width": 256})
        }
        DataType::Extension(_, inner_data_type, _) => serialize_data_type(inner_data_type),
    }
}

fn serialize_field(field: &Field, ipc_field: &IpcField) -> ArrowJsonField {
    let children = match field.data_type() {
        DataType::Union(fields, ..) | DataType::Struct(fields) => fields
            .iter()
            .zip(ipc_field.fields.iter())
            .map(|(field, ipc_field)| serialize_field(field, ipc_field))
            .collect(),
        DataType::Map(field, ..)
        | DataType::FixedSizeList(field, _)
        | DataType::LargeList(field)
        | DataType::List(field) => {
            vec![serialize_field(field, &ipc_field.fields[0])]
        }
        _ => vec![],
    };
    let metadata = serialize_metadata(&field.metadata);

    let dictionary = if let DataType::Dictionary(key_type, _, is_ordered) = field.data_type() {
        use crate::datatypes::IntegerType::*;
        Some(ArrowJsonFieldDictionary {
            id: ipc_field.dictionary_id.unwrap(),
            index_type: IntegerType {
                name: "".to_string(),
                bit_width: match key_type {
                    Int8 | UInt8 => 8,
                    Int16 | UInt16 => 16,
                    Int32 | UInt32 => 32,
                    Int64 | UInt64 => 64,
                },
                is_signed: match key_type {
                    Int8 | Int16 | Int32 | Int64 => true,
                    UInt8 | UInt16 | UInt32 | UInt64 => false,
                },
            },
            is_ordered: *is_ordered,
        })
    } else {
        None
    };

    ArrowJsonField {
        name: field.name.clone(),
        field_type: serialize_data_type(field.data_type()),
        nullable: field.is_nullable,
        children,
        dictionary,
        metadata,
    }
}

/// Serializes a [`Schema`] and associated [`IpcField`] to [`ArrowJsonSchema`].
pub fn serialize_schema(schema: &Schema, ipc_fields: &[IpcField]) -> ArrowJsonSchema {
    ArrowJsonSchema {
        fields: schema
            .fields
            .iter()
            .zip(ipc_fields.iter())
            .map(|(field, ipc_field)| serialize_field(field, ipc_field))
            .collect(),
        metadata: Some(serde_json::to_value(&schema.metadata).unwrap()),
    }
}

fn serialize_metadata(metadata: &Metadata) -> Option<Value> {
    let array = metadata
        .iter()
        .map(|(k, v)| {
            let mut kv_map = Map::new();
            kv_map.insert(k.clone(), Value::String(v.clone()));
            Value::Object(kv_map)
        })
        .collect::<Vec<_>>();

    if !array.is_empty() {
        Some(Value::Array(array))
    } else {
        None
    }
}
