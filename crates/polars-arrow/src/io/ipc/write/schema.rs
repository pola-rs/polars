use arrow_format::ipc::planus::Builder;

use super::super::IpcField;
use crate::datatypes::{
    ArrowDataType, ArrowSchema, Field, IntegerType, IntervalUnit, Metadata, TimeUnit, UnionMode,
};
use crate::io::ipc::endianness::is_native_little_endian;

/// Converts a [ArrowSchema] and [IpcField]s to a flatbuffers-encoded [arrow_format::ipc::Message].
pub fn schema_to_bytes(
    schema: &ArrowSchema,
    ipc_fields: &[IpcField],
    custom_metadata: Option<&Metadata>,
) -> Vec<u8> {
    let schema = serialize_schema(schema, ipc_fields, custom_metadata);

    let message = arrow_format::ipc::Message {
        version: arrow_format::ipc::MetadataVersion::V5,
        header: Some(arrow_format::ipc::MessageHeader::Schema(Box::new(schema))),
        body_length: 0,
        custom_metadata: None, // todo: allow writing custom metadata
    };
    let mut builder = Builder::new();
    let footer_data = builder.finish(&message, None);
    footer_data.to_vec()
}

pub fn serialize_schema(
    schema: &ArrowSchema,
    ipc_fields: &[IpcField],
    custom_schema_metadata: Option<&Metadata>,
) -> arrow_format::ipc::Schema {
    let endianness = if is_native_little_endian() {
        arrow_format::ipc::Endianness::Little
    } else {
        arrow_format::ipc::Endianness::Big
    };

    let fields = schema
        .iter_values()
        .zip(ipc_fields.iter())
        .map(|(field, ipc_field)| serialize_field(field, ipc_field))
        .collect::<Vec<_>>();

    let custom_metadata = custom_schema_metadata.and_then(|custom_meta| {
        let as_kv = custom_meta
            .iter()
            .map(|(key, val)| key_value(key.clone().into_string(), val.clone().into_string()))
            .collect::<Vec<_>>();
        (!as_kv.is_empty()).then_some(as_kv)
    });

    arrow_format::ipc::Schema {
        endianness,
        fields: Some(fields),
        custom_metadata,
        features: None, // todo add this one
    }
}

fn key_value(key: impl Into<String>, val: impl Into<String>) -> arrow_format::ipc::KeyValue {
    arrow_format::ipc::KeyValue {
        key: Some(key.into()),
        value: Some(val.into()),
    }
}

fn write_metadata(metadata: &Metadata, kv_vec: &mut Vec<arrow_format::ipc::KeyValue>) {
    for (k, v) in metadata {
        if k.as_str() != "ARROW:extension:name" && k.as_str() != "ARROW:extension:metadata" {
            kv_vec.push(key_value(k.clone().into_string(), v.clone().into_string()));
        }
    }
}

fn write_extension(
    name: &str,
    metadata: Option<&str>,
    kv_vec: &mut Vec<arrow_format::ipc::KeyValue>,
) {
    if let Some(metadata) = metadata {
        kv_vec.push(key_value("ARROW:extension:metadata".to_string(), metadata));
    }

    kv_vec.push(key_value("ARROW:extension:name".to_string(), name));
}

/// Create an IPC Field from an Arrow Field
pub(crate) fn serialize_field(field: &Field, ipc_field: &IpcField) -> arrow_format::ipc::Field {
    // custom metadata.
    let mut kv_vec = vec![];
    if let ArrowDataType::Extension(ext) = field.dtype() {
        write_extension(
            &ext.name,
            ext.metadata.as_ref().map(|x| x.as_str()),
            &mut kv_vec,
        );
    }

    let type_ = serialize_type(field.dtype());
    let children = serialize_children(field.dtype(), ipc_field);

    let dictionary = if let ArrowDataType::Dictionary(index_type, inner, is_ordered) = field.dtype()
    {
        if let ArrowDataType::Extension(ext) = inner.as_ref() {
            write_extension(
                ext.name.as_str(),
                ext.metadata.as_ref().map(|x| x.as_str()),
                &mut kv_vec,
            );
        }
        Some(serialize_dictionary(
            index_type,
            ipc_field
                .dictionary_id
                .expect("All Dictionary types have `dict_id`"),
            *is_ordered,
        ))
    } else {
        None
    };

    if let Some(metadata) = &field.metadata {
        write_metadata(metadata, &mut kv_vec);
    }

    let custom_metadata = if !kv_vec.is_empty() {
        Some(kv_vec)
    } else {
        None
    };

    arrow_format::ipc::Field {
        name: Some(field.name.to_string()),
        nullable: field.is_nullable,
        type_: Some(type_),
        dictionary: dictionary.map(Box::new),
        children: Some(children),
        custom_metadata,
    }
}

fn serialize_time_unit(unit: &TimeUnit) -> arrow_format::ipc::TimeUnit {
    match unit {
        TimeUnit::Second => arrow_format::ipc::TimeUnit::Second,
        TimeUnit::Millisecond => arrow_format::ipc::TimeUnit::Millisecond,
        TimeUnit::Microsecond => arrow_format::ipc::TimeUnit::Microsecond,
        TimeUnit::Nanosecond => arrow_format::ipc::TimeUnit::Nanosecond,
    }
}

fn serialize_type(dtype: &ArrowDataType) -> arrow_format::ipc::Type {
    use ArrowDataType::*;
    use arrow_format::ipc;
    match dtype {
        Null => ipc::Type::Null(Box::new(ipc::Null {})),
        Boolean => ipc::Type::Bool(Box::new(ipc::Bool {})),
        UInt8 => ipc::Type::Int(Box::new(ipc::Int {
            bit_width: 8,
            is_signed: false,
        })),
        UInt16 => ipc::Type::Int(Box::new(ipc::Int {
            bit_width: 16,
            is_signed: false,
        })),
        UInt32 => ipc::Type::Int(Box::new(ipc::Int {
            bit_width: 32,
            is_signed: false,
        })),
        UInt64 => ipc::Type::Int(Box::new(ipc::Int {
            bit_width: 64,
            is_signed: false,
        })),
        Int8 => ipc::Type::Int(Box::new(ipc::Int {
            bit_width: 8,
            is_signed: true,
        })),
        Int16 => ipc::Type::Int(Box::new(ipc::Int {
            bit_width: 16,
            is_signed: true,
        })),
        Int32 => ipc::Type::Int(Box::new(ipc::Int {
            bit_width: 32,
            is_signed: true,
        })),
        Int64 => ipc::Type::Int(Box::new(ipc::Int {
            bit_width: 64,
            is_signed: true,
        })),
        Int128 => ipc::Type::Int(Box::new(ipc::Int {
            bit_width: 128,
            is_signed: true,
        })),
        Float16 => ipc::Type::FloatingPoint(Box::new(ipc::FloatingPoint {
            precision: ipc::Precision::Half,
        })),
        Float32 => ipc::Type::FloatingPoint(Box::new(ipc::FloatingPoint {
            precision: ipc::Precision::Single,
        })),
        Float64 => ipc::Type::FloatingPoint(Box::new(ipc::FloatingPoint {
            precision: ipc::Precision::Double,
        })),
        Decimal(precision, scale) => ipc::Type::Decimal(Box::new(ipc::Decimal {
            precision: *precision as i32,
            scale: *scale as i32,
            bit_width: 128,
        })),
        Decimal32(precision, scale) => ipc::Type::Decimal(Box::new(ipc::Decimal {
            precision: *precision as i32,
            scale: *scale as i32,
            bit_width: 32,
        })),
        Decimal64(precision, scale) => ipc::Type::Decimal(Box::new(ipc::Decimal {
            precision: *precision as i32,
            scale: *scale as i32,
            bit_width: 64,
        })),
        Decimal256(precision, scale) => ipc::Type::Decimal(Box::new(ipc::Decimal {
            precision: *precision as i32,
            scale: *scale as i32,
            bit_width: 256,
        })),
        Binary => ipc::Type::Binary(Box::new(ipc::Binary {})),
        LargeBinary => ipc::Type::LargeBinary(Box::new(ipc::LargeBinary {})),
        Utf8 => ipc::Type::Utf8(Box::new(ipc::Utf8 {})),
        LargeUtf8 => ipc::Type::LargeUtf8(Box::new(ipc::LargeUtf8 {})),
        FixedSizeBinary(size) => ipc::Type::FixedSizeBinary(Box::new(ipc::FixedSizeBinary {
            byte_width: *size as i32,
        })),
        Date32 => ipc::Type::Date(Box::new(ipc::Date {
            unit: ipc::DateUnit::Day,
        })),
        Date64 => ipc::Type::Date(Box::new(ipc::Date {
            unit: ipc::DateUnit::Millisecond,
        })),
        Duration(unit) => ipc::Type::Duration(Box::new(ipc::Duration {
            unit: serialize_time_unit(unit),
        })),
        Time32(unit) => ipc::Type::Time(Box::new(ipc::Time {
            unit: serialize_time_unit(unit),
            bit_width: 32,
        })),
        Time64(unit) => ipc::Type::Time(Box::new(ipc::Time {
            unit: serialize_time_unit(unit),
            bit_width: 64,
        })),
        Timestamp(unit, tz) => ipc::Type::Timestamp(Box::new(ipc::Timestamp {
            unit: serialize_time_unit(unit),
            timezone: tz.as_ref().map(|x| x.to_string()),
        })),
        Interval(unit) => ipc::Type::Interval(Box::new(ipc::Interval {
            unit: match unit {
                IntervalUnit::YearMonth => ipc::IntervalUnit::YearMonth,
                IntervalUnit::DayTime => ipc::IntervalUnit::DayTime,
                IntervalUnit::MonthDayNano => ipc::IntervalUnit::MonthDayNano,
            },
        })),
        List(_) => ipc::Type::List(Box::new(ipc::List {})),
        LargeList(_) => ipc::Type::LargeList(Box::new(ipc::LargeList {})),
        FixedSizeList(_, size) => ipc::Type::FixedSizeList(Box::new(ipc::FixedSizeList {
            list_size: *size as i32,
        })),
        Union(u) => ipc::Type::Union(Box::new(ipc::Union {
            mode: match u.mode {
                UnionMode::Dense => ipc::UnionMode::Dense,
                UnionMode::Sparse => ipc::UnionMode::Sparse,
            },
            type_ids: u.ids.clone(),
        })),
        Map(_, keys_sorted) => ipc::Type::Map(Box::new(ipc::Map {
            keys_sorted: *keys_sorted,
        })),
        Struct(_) => ipc::Type::Struct(Box::new(ipc::Struct {})),
        Dictionary(_, v, _) => serialize_type(v),
        Extension(ext) => serialize_type(&ext.inner),
        Utf8View => ipc::Type::Utf8View(Box::new(ipc::Utf8View {})),
        BinaryView => ipc::Type::BinaryView(Box::new(ipc::BinaryView {})),
        Unknown => unimplemented!(),
    }
}

fn serialize_children(
    dtype: &ArrowDataType,
    ipc_field: &IpcField,
) -> Vec<arrow_format::ipc::Field> {
    use ArrowDataType::*;
    match dtype {
        Null
        | Boolean
        | Int8
        | Int16
        | Int32
        | Int64
        | UInt8
        | UInt16
        | UInt32
        | UInt64
        | Int128
        | Float16
        | Float32
        | Float64
        | Timestamp(_, _)
        | Date32
        | Date64
        | Time32(_)
        | Time64(_)
        | Duration(_)
        | Interval(_)
        | Binary
        | FixedSizeBinary(_)
        | LargeBinary
        | Utf8
        | LargeUtf8
        | Utf8View
        | BinaryView
        | Decimal(_, _)
        | Decimal32(_, _)
        | Decimal64(_, _)
        | Decimal256(_, _) => vec![],
        FixedSizeList(inner, _) | LargeList(inner) | List(inner) | Map(inner, _) => {
            vec![serialize_field(inner, &ipc_field.fields[0])]
        },
        Struct(fields) => fields
            .iter()
            .zip(ipc_field.fields.iter())
            .map(|(field, ipc)| serialize_field(field, ipc))
            .collect(),
        Union(u) => u
            .fields
            .iter()
            .zip(ipc_field.fields.iter())
            .map(|(field, ipc)| serialize_field(field, ipc))
            .collect(),
        Dictionary(_, inner, _) => serialize_children(inner, ipc_field),
        Extension(ext) => serialize_children(&ext.inner, ipc_field),
        Unknown => unimplemented!(),
    }
}

/// Create an IPC dictionary encoding
pub(crate) fn serialize_dictionary(
    index_type: &IntegerType,
    dict_id: i64,
    dict_is_ordered: bool,
) -> arrow_format::ipc::DictionaryEncoding {
    use IntegerType::*;
    let is_signed = match index_type {
        Int8 | Int16 | Int32 | Int64 | Int128 => true,
        UInt8 | UInt16 | UInt32 | UInt64 => false,
    };

    let bit_width = match index_type {
        Int8 | UInt8 => 8,
        Int16 | UInt16 => 16,
        Int32 | UInt32 => 32,
        Int64 | UInt64 => 64,
        Int128 => 128,
    };

    let index_type = arrow_format::ipc::Int {
        bit_width,
        is_signed,
    };

    arrow_format::ipc::DictionaryEncoding {
        id: dict_id,
        index_type: Some(Box::new(index_type)),
        is_ordered: dict_is_ordered,
        dictionary_kind: arrow_format::ipc::DictionaryKind::DenseArray,
    }
}
