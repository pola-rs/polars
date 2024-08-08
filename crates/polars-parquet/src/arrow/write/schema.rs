use arrow::datatypes::{ArrowDataType, ArrowSchema, Field, TimeUnit};
use arrow::io::ipc::write::{default_ipc_fields, schema_to_bytes};
use base64::engine::general_purpose;
use base64::Engine as _;
use polars_error::{polars_bail, PolarsResult};

use super::super::ARROW_SCHEMA_META_KEY;
use crate::arrow::write::decimal_length_from_precision;
use crate::parquet::metadata::KeyValue;
use crate::parquet::schema::types::{
    GroupConvertedType, GroupLogicalType, IntegerType, ParquetType, PhysicalType,
    PrimitiveConvertedType, PrimitiveLogicalType, TimeUnit as ParquetTimeUnit,
};
use crate::parquet::schema::Repetition;

fn convert_field(field: Field) -> Field {
    Field {
        name: field.name,
        data_type: convert_data_type(field.data_type),
        is_nullable: field.is_nullable,
        metadata: field.metadata,
    }
}

fn convert_data_type(data_type: ArrowDataType) -> ArrowDataType {
    use ArrowDataType as D;
    match data_type {
        D::LargeList(field) => D::LargeList(Box::new(convert_field(*field))),
        D::Struct(mut fields) => {
            for field in &mut fields {
                *field = convert_field(std::mem::take(field))
            }
            D::Struct(fields)
        },
        D::BinaryView => D::LargeBinary,
        D::Utf8View => D::LargeUtf8,
        D::Dictionary(it, data_type, sorted) => {
            let dtype = convert_data_type(*data_type);
            D::Dictionary(it, Box::new(dtype), sorted)
        },
        D::Extension(name, data_type, metadata) => {
            let data_type = convert_data_type(*data_type);
            D::Extension(name, Box::new(data_type), metadata)
        },
        dt => dt,
    }
}

pub fn schema_to_metadata_key(schema: &ArrowSchema) -> KeyValue {
    // Convert schema until more arrow readers are aware of binview
    let serialized_schema = if schema.fields.iter().any(|field| field.data_type.is_view()) {
        let fields = schema
            .fields
            .iter()
            .map(|field| convert_field(field.clone()))
            .collect::<Vec<_>>();
        let schema = ArrowSchema::from(fields);
        schema_to_bytes(&schema, &default_ipc_fields(&schema.fields))
    } else {
        schema_to_bytes(schema, &default_ipc_fields(&schema.fields))
    };

    // manually prepending the length to the schema as arrow uses the legacy IPC format
    // TODO: change after addressing ARROW-9777
    let schema_len = serialized_schema.len();
    let mut len_prefix_schema = Vec::with_capacity(schema_len + 8);
    len_prefix_schema.extend_from_slice(&[255u8, 255, 255, 255]);
    len_prefix_schema.extend_from_slice(&(schema_len as u32).to_le_bytes());
    len_prefix_schema.extend_from_slice(&serialized_schema);

    let encoded = general_purpose::STANDARD.encode(&len_prefix_schema);

    KeyValue {
        key: ARROW_SCHEMA_META_KEY.to_string(),
        value: Some(encoded),
    }
}

/// Creates a [`ParquetType`] from a [`Field`].
pub fn to_parquet_type(field: &Field) -> PolarsResult<ParquetType> {
    let name = field.name.clone();
    let repetition = if field.is_nullable {
        Repetition::Optional
    } else {
        Repetition::Required
    };
    // create type from field
    match field.data_type().to_logical_type() {
        ArrowDataType::Null => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Int32,
            repetition,
            None,
            Some(PrimitiveLogicalType::Unknown),
            None,
        )?),
        ArrowDataType::Boolean => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Boolean,
            repetition,
            None,
            None,
            None,
        )?),
        ArrowDataType::Int32 => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Int32,
            repetition,
            None,
            None,
            None,
        )?),
        // ArrowDataType::Duration(_) has no parquet representation => do not apply any logical type
        ArrowDataType::Int64 | ArrowDataType::Duration(_) => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Int64,
            repetition,
            None,
            None,
            None,
        )?),
        // no natural representation in parquet; leave it as is.
        // arrow consumers MAY use the arrow schema in the metadata to parse them.
        ArrowDataType::Date64 => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Int64,
            repetition,
            None,
            None,
            None,
        )?),
        ArrowDataType::Float32 => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Float,
            repetition,
            None,
            None,
            None,
        )?),
        ArrowDataType::Float64 => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Double,
            repetition,
            None,
            None,
            None,
        )?),
        ArrowDataType::Binary | ArrowDataType::LargeBinary | ArrowDataType::BinaryView => {
            Ok(ParquetType::try_from_primitive(
                name,
                PhysicalType::ByteArray,
                repetition,
                None,
                None,
                None,
            )?)
        },
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => {
            Ok(ParquetType::try_from_primitive(
                name,
                PhysicalType::ByteArray,
                repetition,
                Some(PrimitiveConvertedType::Utf8),
                Some(PrimitiveLogicalType::String),
                None,
            )?)
        },
        ArrowDataType::Date32 => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Int32,
            repetition,
            Some(PrimitiveConvertedType::Date),
            Some(PrimitiveLogicalType::Date),
            None,
        )?),
        ArrowDataType::Int8 => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Int32,
            repetition,
            Some(PrimitiveConvertedType::Int8),
            Some(PrimitiveLogicalType::Integer(IntegerType::Int8)),
            None,
        )?),
        ArrowDataType::Int16 => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Int32,
            repetition,
            Some(PrimitiveConvertedType::Int16),
            Some(PrimitiveLogicalType::Integer(IntegerType::Int16)),
            None,
        )?),
        ArrowDataType::UInt8 => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Int32,
            repetition,
            Some(PrimitiveConvertedType::Uint8),
            Some(PrimitiveLogicalType::Integer(IntegerType::UInt8)),
            None,
        )?),
        ArrowDataType::UInt16 => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Int32,
            repetition,
            Some(PrimitiveConvertedType::Uint16),
            Some(PrimitiveLogicalType::Integer(IntegerType::UInt16)),
            None,
        )?),
        ArrowDataType::UInt32 => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Int32,
            repetition,
            Some(PrimitiveConvertedType::Uint32),
            Some(PrimitiveLogicalType::Integer(IntegerType::UInt32)),
            None,
        )?),
        ArrowDataType::UInt64 => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Int64,
            repetition,
            Some(PrimitiveConvertedType::Uint64),
            Some(PrimitiveLogicalType::Integer(IntegerType::UInt64)),
            None,
        )?),
        // no natural representation in parquet; leave it as is.
        // arrow consumers MAY use the arrow schema in the metadata to parse them.
        ArrowDataType::Timestamp(TimeUnit::Second, _) => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Int64,
            repetition,
            None,
            None,
            None,
        )?),
        ArrowDataType::Timestamp(time_unit, zone) => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Int64,
            repetition,
            None,
            Some(PrimitiveLogicalType::Timestamp {
                is_adjusted_to_utc: matches!(zone, Some(z) if !z.as_str().is_empty()),
                unit: match time_unit {
                    TimeUnit::Second => unreachable!(),
                    TimeUnit::Millisecond => ParquetTimeUnit::Milliseconds,
                    TimeUnit::Microsecond => ParquetTimeUnit::Microseconds,
                    TimeUnit::Nanosecond => ParquetTimeUnit::Nanoseconds,
                },
            }),
            None,
        )?),
        // no natural representation in parquet; leave it as is.
        // arrow consumers MAY use the arrow schema in the metadata to parse them.
        ArrowDataType::Time32(TimeUnit::Second) => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Int32,
            repetition,
            None,
            None,
            None,
        )?),
        ArrowDataType::Time32(TimeUnit::Millisecond) => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Int32,
            repetition,
            Some(PrimitiveConvertedType::TimeMillis),
            Some(PrimitiveLogicalType::Time {
                is_adjusted_to_utc: false,
                unit: ParquetTimeUnit::Milliseconds,
            }),
            None,
        )?),
        ArrowDataType::Time64(time_unit) => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::Int64,
            repetition,
            match time_unit {
                TimeUnit::Microsecond => Some(PrimitiveConvertedType::TimeMicros),
                TimeUnit::Nanosecond => None,
                _ => unreachable!(),
            },
            Some(PrimitiveLogicalType::Time {
                is_adjusted_to_utc: false,
                unit: match time_unit {
                    TimeUnit::Microsecond => ParquetTimeUnit::Microseconds,
                    TimeUnit::Nanosecond => ParquetTimeUnit::Nanoseconds,
                    _ => unreachable!(),
                },
            }),
            None,
        )?),
        ArrowDataType::Struct(fields) => {
            if fields.is_empty() {
                polars_bail!(InvalidOperation:
                    "Parquet does not support writing empty structs".to_string(),
                )
            }
            // recursively convert children to types/nodes
            let fields = fields
                .iter()
                .map(to_parquet_type)
                .collect::<PolarsResult<Vec<_>>>()?;
            Ok(ParquetType::from_group(
                name, repetition, None, None, fields, None,
            ))
        },
        ArrowDataType::Dictionary(_, value, _) => {
            let dict_field = Field::new(name.as_str(), value.as_ref().clone(), field.is_nullable);
            to_parquet_type(&dict_field)
        },
        ArrowDataType::FixedSizeBinary(size) => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::FixedLenByteArray(*size),
            repetition,
            None,
            None,
            None,
        )?),
        ArrowDataType::Decimal(precision, scale) => {
            let precision = *precision;
            let scale = *scale;
            let logical_type = Some(PrimitiveLogicalType::Decimal(precision, scale));

            let physical_type = if precision <= 9 {
                PhysicalType::Int32
            } else if precision <= 18 {
                PhysicalType::Int64
            } else {
                let len = decimal_length_from_precision(precision);
                PhysicalType::FixedLenByteArray(len)
            };
            Ok(ParquetType::try_from_primitive(
                name,
                physical_type,
                repetition,
                Some(PrimitiveConvertedType::Decimal(precision, scale)),
                logical_type,
                None,
            )?)
        },
        ArrowDataType::Decimal256(precision, scale) => {
            let precision = *precision;
            let scale = *scale;
            let logical_type = Some(PrimitiveLogicalType::Decimal(precision, scale));

            if precision <= 9 {
                Ok(ParquetType::try_from_primitive(
                    name,
                    PhysicalType::Int32,
                    repetition,
                    Some(PrimitiveConvertedType::Decimal(precision, scale)),
                    logical_type,
                    None,
                )?)
            } else if precision <= 18 {
                Ok(ParquetType::try_from_primitive(
                    name,
                    PhysicalType::Int64,
                    repetition,
                    Some(PrimitiveConvertedType::Decimal(precision, scale)),
                    logical_type,
                    None,
                )?)
            } else if precision <= 38 {
                let len = decimal_length_from_precision(precision);
                Ok(ParquetType::try_from_primitive(
                    name,
                    PhysicalType::FixedLenByteArray(len),
                    repetition,
                    Some(PrimitiveConvertedType::Decimal(precision, scale)),
                    logical_type,
                    None,
                )?)
            } else {
                Ok(ParquetType::try_from_primitive(
                    name,
                    PhysicalType::FixedLenByteArray(32),
                    repetition,
                    None,
                    None,
                    None,
                )?)
            }
        },
        ArrowDataType::Interval(_) => Ok(ParquetType::try_from_primitive(
            name,
            PhysicalType::FixedLenByteArray(12),
            repetition,
            Some(PrimitiveConvertedType::Interval),
            None,
            None,
        )?),
        ArrowDataType::List(f)
        | ArrowDataType::FixedSizeList(f, _)
        | ArrowDataType::LargeList(f) => {
            let mut f = f.clone();
            f.name = "element".to_string();

            Ok(ParquetType::from_group(
                name,
                repetition,
                Some(GroupConvertedType::List),
                Some(GroupLogicalType::List),
                vec![ParquetType::from_group(
                    "list".to_string(),
                    Repetition::Repeated,
                    None,
                    None,
                    vec![to_parquet_type(&f)?],
                    None,
                )],
                None,
            ))
        },
        ArrowDataType::Map(f, _) => Ok(ParquetType::from_group(
            name,
            repetition,
            Some(GroupConvertedType::Map),
            Some(GroupLogicalType::Map),
            vec![ParquetType::from_group(
                "map".to_string(),
                Repetition::Repeated,
                None,
                None,
                vec![to_parquet_type(f)?],
                None,
            )],
            None,
        )),
        other => polars_bail!(nyi = "Writing the data type {other:?} is not yet implemented"),
    }
}
