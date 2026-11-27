//! Iceberg types → Arrow types.
//!
//! Two mappings are used:
//! * [`table_field`]: the table schema sent to the host, with `PARQUET:field_id` metadata. It
//!   matches PyIceberg's `schema_to_pyarrow`, which the Python resolver uses for the same purpose.
//! * [`value_dtype`]: the Arrow types of values built by the plugin (statistics, partition
//!   constants, defaults), using the types Polars stores natively (views, nanosecond time).
use polars_arrow::datatypes::{ArrowDataType, ExtensionType, Field, Metadata, TimeUnit};

use crate::iceberg::spec::{NestedField, PrimitiveType, Schema, Type};

pub const FIELD_ID_KEY: &str = "PARQUET:field_id";
const UUID_EXTENSION_NAME: &str = "arrow.uuid";
const UTC: &str = "UTC";

pub fn table_fields(schema: &Schema) -> Vec<Field> {
    schema.fields.iter().map(table_field).collect()
}

pub fn table_field(field: &NestedField) -> Field {
    with_id(
        Field::new(
            field.name.as_str().into(),
            table_dtype(&field.field_type),
            !field.required,
        ),
        field.id,
    )
}

fn with_id(field: Field, id: i32) -> Field {
    let mut metadata = Metadata::new();
    metadata.insert(FIELD_ID_KEY.into(), id.to_string().into());
    field.with_metadata(metadata)
}

fn table_dtype(ty: &Type) -> ArrowDataType {
    match ty {
        Type::Primitive(p) => match p {
            PrimitiveType::String => ArrowDataType::LargeUtf8,
            PrimitiveType::Binary => ArrowDataType::LargeBinary,
            PrimitiveType::Uuid => ArrowDataType::Extension(Box::new(ExtensionType {
                name: UUID_EXTENSION_NAME.into(),
                inner: ArrowDataType::FixedSizeBinary(16),
                metadata: None,
            })),
            PrimitiveType::Fixed(n) => ArrowDataType::FixedSizeBinary(*n as usize),
            PrimitiveType::Time => ArrowDataType::Time64(TimeUnit::Microsecond),
            p => primitive_dtype(p),
        },
        Type::Struct(fields) => ArrowDataType::Struct(fields.iter().map(table_field).collect()),
        Type::List {
            element_id,
            element_required,
            element,
        } => ArrowDataType::LargeList(Box::new(with_id(
            Field::new("element".into(), table_dtype(element), !element_required),
            *element_id,
        ))),
        Type::Map {
            key_id,
            key,
            value_id,
            value_required,
            value,
        } => ArrowDataType::Map(
            Box::new(Field::new(
                "entries".into(),
                ArrowDataType::Struct(vec![
                    with_id(Field::new("key".into(), table_dtype(key), false), *key_id),
                    with_id(
                        Field::new("value".into(), table_dtype(value), !value_required),
                        *value_id,
                    ),
                ]),
                false,
            )),
            false,
        ),
    }
}

/// Arrow type of values of an Iceberg type, as built by the plugin.
pub fn value_dtype(ty: &Type) -> ArrowDataType {
    match ty {
        Type::Primitive(p) => match p {
            PrimitiveType::String => ArrowDataType::Utf8View,
            PrimitiveType::Binary | PrimitiveType::Uuid | PrimitiveType::Fixed(_) => {
                ArrowDataType::BinaryView
            },
            PrimitiveType::Time => ArrowDataType::Time64(TimeUnit::Nanosecond),
            p => primitive_dtype(p),
        },
        Type::Struct(fields) => ArrowDataType::Struct(
            fields
                .iter()
                .map(|f| Field::new(f.name.as_str().into(), value_dtype(&f.field_type), true))
                .collect(),
        ),
        Type::List { element, .. } => ArrowDataType::LargeList(Box::new(Field::new(
            "item".into(),
            value_dtype(element),
            true,
        ))),
        Type::Map { key, value, .. } => ArrowDataType::Map(
            Box::new(Field::new(
                "entries".into(),
                ArrowDataType::Struct(vec![
                    Field::new("key".into(), value_dtype(key), false),
                    Field::new("value".into(), value_dtype(value), true),
                ]),
                false,
            )),
            false,
        ),
    }
}

/// Null counts in the statistics: one count per leaf for structs, one count otherwise.
pub fn null_count_dtype(ty: &Type) -> ArrowDataType {
    match ty {
        Type::Struct(fields) => ArrowDataType::Struct(
            fields
                .iter()
                .map(|f| {
                    Field::new(
                        f.name.as_str().into(),
                        null_count_dtype(&f.field_type),
                        true,
                    )
                })
                .collect(),
        ),
        _ => ArrowDataType::UInt64,
    }
}

fn primitive_dtype(p: &PrimitiveType) -> ArrowDataType {
    use PrimitiveType as P;
    match p {
        P::Boolean => ArrowDataType::Boolean,
        P::Int => ArrowDataType::Int32,
        P::Long => ArrowDataType::Int64,
        P::Float => ArrowDataType::Float32,
        P::Double => ArrowDataType::Float64,
        P::Decimal { precision, scale } => {
            ArrowDataType::Decimal(*precision as usize, *scale as usize)
        },
        P::Date => ArrowDataType::Date32,
        P::Time => ArrowDataType::Time64(TimeUnit::Nanosecond),
        P::Timestamp => ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
        P::Timestamptz => ArrowDataType::Timestamp(TimeUnit::Microsecond, Some(UTC.into())),
        P::TimestampNs => ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),
        P::TimestamptzNs => ArrowDataType::Timestamp(TimeUnit::Nanosecond, Some(UTC.into())),
        P::String => ArrowDataType::Utf8View,
        P::Binary | P::Uuid | P::Fixed(_) => ArrowDataType::BinaryView,
        P::Unknown => ArrowDataType::Null,
    }
}
