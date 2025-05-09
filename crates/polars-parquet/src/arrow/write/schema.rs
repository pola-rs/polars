use std::borrow::Cow;
use std::sync::Arc;

use arrow::datatypes::{ArrowDataType, ArrowSchema, ExtensionType, Field, TimeUnit};
use arrow::io::ipc::write::{default_ipc_fields, schema_to_bytes};
use base64::Engine as _;
use base64::engine::general_purpose;
use polars_error::{PolarsResult, polars_bail};
use polars_utils::pl_str::PlSmallStr;

use super::super::ARROW_SCHEMA_META_KEY;
use super::ColumnWriteOptions;
use crate::arrow::write::decimal_length_from_precision;
use crate::parquet::metadata::KeyValue;
use crate::parquet::schema::Repetition;
use crate::parquet::schema::types::{
    GroupConvertedType, GroupLogicalType, IntegerType, ParquetType, PhysicalType,
    PrimitiveConvertedType, PrimitiveLogicalType, TimeUnit as ParquetTimeUnit,
};
use crate::write::ChildWriteOptions;

fn convert_field(field: Field) -> Field {
    Field {
        name: field.name,
        dtype: convert_dtype(field.dtype),
        is_nullable: field.is_nullable,
        metadata: field.metadata,
    }
}

fn convert_dtype(dtype: ArrowDataType) -> ArrowDataType {
    use ArrowDataType as D;
    match dtype {
        D::LargeList(field) => D::LargeList(Box::new(convert_field(*field))),
        D::Struct(mut fields) => {
            for field in &mut fields {
                *field = convert_field(std::mem::take(field))
            }
            D::Struct(fields)
        },
        D::BinaryView => D::LargeBinary,
        D::Utf8View => D::LargeUtf8,
        D::Dictionary(it, dtype, sorted) => {
            let dtype = convert_dtype(*dtype);
            D::Dictionary(it, Box::new(dtype), sorted)
        },
        D::Extension(ext) => {
            let dtype = convert_dtype(ext.inner);
            D::Extension(Box::new(ExtensionType {
                inner: dtype,
                ..*ext
            }))
        },
        dt => dt,
    }
}

fn insert_field_metadata(field: &mut Cow<Field>, options: &ColumnWriteOptions) {
    if !options.metadata.is_empty() {
        let field = field.to_mut();
        let mut metadata = field.metadata.as_deref().cloned().unwrap_or_default();

        for kv in &options.metadata {
            metadata.insert(
                kv.key.as_str().into(),
                kv.value.as_deref().unwrap_or_default().into(),
            );
        }
        field.metadata = Some(Arc::new(metadata));
    }

    use ArrowDataType as D;
    match field.dtype() {
        D::Struct(f) => {
            let ChildWriteOptions::Struct(o) = &options.children else {
                unreachable!();
            };

            let mut new_fields = Vec::new();
            for (i, (child_field, child_options)) in f.iter().zip(o.children.as_slice()).enumerate()
            {
                let mut child_field = Cow::Borrowed(child_field);
                insert_field_metadata(&mut child_field, child_options);

                if let Cow::Owned(child_field) = child_field {
                    new_fields.reserve(f.len());
                    new_fields.extend(f[..i].iter().cloned());
                    new_fields.push(child_field);
                    break;
                }
            }

            if new_fields.is_empty() {
                return;
            }

            new_fields.extend(
                f[new_fields.len()..]
                    .iter()
                    .zip(&o.children[new_fields.len()..])
                    .map(|(child_field, child_options)| {
                        let mut child_field = Cow::Borrowed(child_field);
                        insert_field_metadata(&mut child_field, child_options);
                        child_field.into_owned()
                    }),
            );
            field
                .to_mut()
                .map_dtype_mut(|dtype| *dtype = D::Struct(new_fields));
        },
        D::List(f) | D::FixedSizeList(f, _) | D::LargeList(f) => {
            let ChildWriteOptions::ListLike(o) = &options.children else {
                unreachable!();
            };

            let mut child_field = Cow::Borrowed(f.as_ref());
            insert_field_metadata(&mut child_field, &o.child);

            if let Cow::Owned(child_field) = child_field {
                let child_field = Box::new(child_field);
                field.to_mut().map_dtype_mut(|dtype| {
                    *dtype = match dtype {
                        D::List(_) => D::List(child_field),
                        D::LargeList(_) => D::LargeList(child_field),
                        D::FixedSizeList(_, width) => D::FixedSizeList(child_field, *width),
                        _ => unreachable!(),
                    }
                });
            }
        },
        _ => {},
    }
}

pub fn schema_to_metadata_key(schema: &ArrowSchema, options: &[ColumnWriteOptions]) -> KeyValue {
    let mut schema_mut = None;
    for (f, options) in schema.iter_values().zip(options) {
        let mut field = Cow::Borrowed(f);
        insert_field_metadata(&mut field, options);

        if let Cow::Owned(field) = field {
            let schema_mut = schema_mut.get_or_insert_with(|| schema.clone());
            *schema_mut.get_mut(f.name.as_str()).unwrap() = field;
        }
    }

    let mut schema = schema;
    if let Some(schema_mut) = &schema_mut {
        schema = schema_mut;
    }

    // Convert schema until more arrow readers are aware of binview
    let serialized_schema = if schema.iter_values().any(|field| field.dtype.is_view()) {
        let schema = schema
            .iter_values()
            .map(|field| convert_field(field.clone()))
            .map(|x| (x.name.clone(), x))
            .collect();
        schema_to_bytes(&schema, &default_ipc_fields(schema.iter_values()), None)
    } else {
        schema_to_bytes(schema, &default_ipc_fields(schema.iter_values()), None)
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
pub fn to_parquet_type(field: &Field, options: &ColumnWriteOptions) -> PolarsResult<ParquetType> {
    let name = field.name.clone();
    let repetition = if field.is_nullable {
        Repetition::Optional
    } else {
        Repetition::Required
    };

    let field_id = options.field_id;

    // create type from field
    let (physical_type, primitive_converted_type, primitive_logical_type) = match field
        .dtype()
        .to_logical_type()
    {
        ArrowDataType::Null => (
            PhysicalType::Int32,
            None,
            Some(PrimitiveLogicalType::Unknown),
        ),
        ArrowDataType::Boolean => (PhysicalType::Boolean, None, None),
        ArrowDataType::Int32 => (PhysicalType::Int32, None, None),
        // ArrowDataType::Duration(_) has no parquet representation => do not apply any logical type
        ArrowDataType::Int64 | ArrowDataType::Duration(_) => (PhysicalType::Int64, None, None),
        // no natural representation in parquet; leave it as is.
        // arrow consumers MAY use the arrow schema in the metadata to parse them.
        ArrowDataType::Date64 => (PhysicalType::Int64, None, None),
        ArrowDataType::Float32 => (PhysicalType::Float, None, None),
        ArrowDataType::Float64 => (PhysicalType::Double, None, None),
        ArrowDataType::Binary | ArrowDataType::LargeBinary | ArrowDataType::BinaryView => {
            (PhysicalType::ByteArray, None, None)
        },
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => (
            PhysicalType::ByteArray,
            Some(PrimitiveConvertedType::Utf8),
            Some(PrimitiveLogicalType::String),
        ),
        ArrowDataType::Date32 => (
            PhysicalType::Int32,
            Some(PrimitiveConvertedType::Date),
            Some(PrimitiveLogicalType::Date),
        ),
        ArrowDataType::Int8 => (
            PhysicalType::Int32,
            Some(PrimitiveConvertedType::Int8),
            Some(PrimitiveLogicalType::Integer(IntegerType::Int8)),
        ),
        ArrowDataType::Int16 => (
            PhysicalType::Int32,
            Some(PrimitiveConvertedType::Int16),
            Some(PrimitiveLogicalType::Integer(IntegerType::Int16)),
        ),
        ArrowDataType::UInt8 => (
            PhysicalType::Int32,
            Some(PrimitiveConvertedType::Uint8),
            Some(PrimitiveLogicalType::Integer(IntegerType::UInt8)),
        ),
        ArrowDataType::UInt16 => (
            PhysicalType::Int32,
            Some(PrimitiveConvertedType::Uint16),
            Some(PrimitiveLogicalType::Integer(IntegerType::UInt16)),
        ),
        ArrowDataType::UInt32 => (
            PhysicalType::Int32,
            Some(PrimitiveConvertedType::Uint32),
            Some(PrimitiveLogicalType::Integer(IntegerType::UInt32)),
        ),
        ArrowDataType::UInt64 => (
            PhysicalType::Int64,
            Some(PrimitiveConvertedType::Uint64),
            Some(PrimitiveLogicalType::Integer(IntegerType::UInt64)),
        ),
        // no natural representation in parquet; leave it as is.
        // arrow consumers MAY use the arrow schema in the metadata to parse them.
        ArrowDataType::Timestamp(TimeUnit::Second, _) => (PhysicalType::Int64, None, None),
        ArrowDataType::Timestamp(time_unit, zone) => (
            PhysicalType::Int64,
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
        ),
        // no natural representation in parquet; leave it as is.
        // arrow consumers MAY use the arrow schema in the metadata to parse them.
        ArrowDataType::Time32(TimeUnit::Second) => (PhysicalType::Int32, None, None),
        ArrowDataType::Time32(TimeUnit::Millisecond) => (
            PhysicalType::Int32,
            Some(PrimitiveConvertedType::TimeMillis),
            Some(PrimitiveLogicalType::Time {
                is_adjusted_to_utc: false,
                unit: ParquetTimeUnit::Milliseconds,
            }),
        ),
        ArrowDataType::Time64(time_unit) => (
            PhysicalType::Int64,
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
        ),
        ArrowDataType::Struct(fields) => {
            if fields.is_empty() {
                polars_bail!(InvalidOperation:
                    "Unable to write struct type with no child field to Parquet. Consider adding a dummy child field.".to_string(),
                )
            }

            let ChildWriteOptions::Struct(struct_write_options) = &options.children else {
                unreachable!();
            };

            assert_eq!(fields.len(), struct_write_options.children.len());

            // recursively convert children to types/nodes
            let fields = fields
                .iter()
                .zip(struct_write_options.children.as_slice())
                .map(|(f, c)| to_parquet_type(f, c))
                .collect::<PolarsResult<Vec<_>>>()?;
            return Ok(ParquetType::from_group(
                name, repetition, None, None, fields, field_id,
            ));
        },
        ArrowDataType::Dictionary(_, value, _) => {
            assert!(!value.is_nested());
            let dict_field = Field::new(name.clone(), value.as_ref().clone(), field.is_nullable);
            return to_parquet_type(&dict_field, options);
        },
        ArrowDataType::FixedSizeBinary(size) => {
            (PhysicalType::FixedLenByteArray(*size), None, None)
        },
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
            (
                physical_type,
                Some(PrimitiveConvertedType::Decimal(precision, scale)),
                logical_type,
            )
        },
        ArrowDataType::Decimal256(precision, scale) => {
            let precision = *precision;
            let scale = *scale;
            let logical_type = Some(PrimitiveLogicalType::Decimal(precision, scale));

            if precision <= 9 {
                (
                    PhysicalType::Int32,
                    Some(PrimitiveConvertedType::Decimal(precision, scale)),
                    logical_type,
                )
            } else if precision <= 18 {
                (
                    PhysicalType::Int64,
                    Some(PrimitiveConvertedType::Decimal(precision, scale)),
                    logical_type,
                )
            } else if precision <= 38 {
                let len = decimal_length_from_precision(precision);
                (
                    PhysicalType::FixedLenByteArray(len),
                    Some(PrimitiveConvertedType::Decimal(precision, scale)),
                    logical_type,
                )
            } else {
                (PhysicalType::FixedLenByteArray(32), None, None)
            }
        },
        ArrowDataType::Interval(_) => (
            PhysicalType::FixedLenByteArray(12),
            Some(PrimitiveConvertedType::Interval),
            None,
        ),
        ArrowDataType::List(f)
        | ArrowDataType::FixedSizeList(f, _)
        | ArrowDataType::LargeList(f) => {
            let mut f = f.clone();
            f.name = PlSmallStr::from_static("element");

            let ChildWriteOptions::ListLike(list_write_options) = &options.children else {
                unreachable!();
            };

            return Ok(ParquetType::from_group(
                name,
                repetition,
                Some(GroupConvertedType::List),
                Some(GroupLogicalType::List),
                vec![ParquetType::from_group(
                    PlSmallStr::from_static("list"),
                    Repetition::Repeated,
                    None,
                    None,
                    vec![to_parquet_type(&f, &list_write_options.child)?],
                    None,
                )],
                field_id,
            ));
        },
        other => polars_bail!(nyi = "Writing the data type {other:?} is not yet implemented"),
    };

    Ok(ParquetType::try_from_primitive(
        name,
        physical_type,
        repetition,
        primitive_converted_type,
        primitive_logical_type,
        field_id,
    )?)
}
