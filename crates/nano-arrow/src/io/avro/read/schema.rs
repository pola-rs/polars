use avro_schema::schema::{Enum, Fixed, Record, Schema as AvroSchema};

use crate::datatypes::*;
use crate::error::{Error, Result};

fn external_props(schema: &AvroSchema) -> Metadata {
    let mut props = Metadata::new();
    match &schema {
        AvroSchema::Record(Record {
            doc: Some(ref doc), ..
        })
        | AvroSchema::Enum(Enum {
            doc: Some(ref doc), ..
        }) => {
            props.insert("avro::doc".to_string(), doc.clone());
        }
        _ => {}
    }
    props
}

/// Infers an [`Schema`] from the root [`Record`].
/// This
pub fn infer_schema(record: &Record) -> Result<Schema> {
    Ok(record
        .fields
        .iter()
        .map(|field| {
            schema_to_field(
                &field.schema,
                Some(&field.name),
                external_props(&field.schema),
            )
        })
        .collect::<Result<Vec<_>>>()?
        .into())
}

fn schema_to_field(schema: &AvroSchema, name: Option<&str>, props: Metadata) -> Result<Field> {
    let mut nullable = false;
    let data_type = match schema {
        AvroSchema::Null => DataType::Null,
        AvroSchema::Boolean => DataType::Boolean,
        AvroSchema::Int(logical) => match logical {
            Some(logical) => match logical {
                avro_schema::schema::IntLogical::Date => DataType::Date32,
                avro_schema::schema::IntLogical::Time => DataType::Time32(TimeUnit::Millisecond),
            },
            None => DataType::Int32,
        },
        AvroSchema::Long(logical) => match logical {
            Some(logical) => match logical {
                avro_schema::schema::LongLogical::Time => DataType::Time64(TimeUnit::Microsecond),
                avro_schema::schema::LongLogical::TimestampMillis => {
                    DataType::Timestamp(TimeUnit::Millisecond, Some("00:00".to_string()))
                }
                avro_schema::schema::LongLogical::TimestampMicros => {
                    DataType::Timestamp(TimeUnit::Microsecond, Some("00:00".to_string()))
                }
                avro_schema::schema::LongLogical::LocalTimestampMillis => {
                    DataType::Timestamp(TimeUnit::Millisecond, None)
                }
                avro_schema::schema::LongLogical::LocalTimestampMicros => {
                    DataType::Timestamp(TimeUnit::Microsecond, None)
                }
            },
            None => DataType::Int64,
        },
        AvroSchema::Float => DataType::Float32,
        AvroSchema::Double => DataType::Float64,
        AvroSchema::Bytes(logical) => match logical {
            Some(logical) => match logical {
                avro_schema::schema::BytesLogical::Decimal(precision, scale) => {
                    DataType::Decimal(*precision, *scale)
                }
            },
            None => DataType::Binary,
        },
        AvroSchema::String(_) => DataType::Utf8,
        AvroSchema::Array(item_schema) => DataType::List(Box::new(schema_to_field(
            item_schema,
            Some("item"), // default name for list items
            Metadata::default(),
        )?)),
        AvroSchema::Map(_) => todo!("Avro maps are mapped to MapArrays"),
        AvroSchema::Union(schemas) => {
            // If there are only two variants and one of them is null, set the other type as the field data type
            let has_nullable = schemas.iter().any(|x| x == &AvroSchema::Null);
            if has_nullable && schemas.len() == 2 {
                nullable = true;
                if let Some(schema) = schemas
                    .iter()
                    .find(|&schema| !matches!(schema, AvroSchema::Null))
                {
                    schema_to_field(schema, None, Metadata::default())?.data_type
                } else {
                    return Err(Error::NotYetImplemented(format!(
                        "Can't read avro union {schema:?}"
                    )));
                }
            } else {
                let fields = schemas
                    .iter()
                    .map(|s| schema_to_field(s, None, Metadata::default()))
                    .collect::<Result<Vec<Field>>>()?;
                DataType::Union(fields, None, UnionMode::Dense)
            }
        }
        AvroSchema::Record(Record { fields, .. }) => {
            let fields = fields
                .iter()
                .map(|field| {
                    let mut props = Metadata::new();
                    if let Some(doc) = &field.doc {
                        props.insert("avro::doc".to_string(), doc.clone());
                    }
                    schema_to_field(&field.schema, Some(&field.name), props)
                })
                .collect::<Result<_>>()?;
            DataType::Struct(fields)
        }
        AvroSchema::Enum { .. } => {
            return Ok(Field::new(
                name.unwrap_or_default(),
                DataType::Dictionary(IntegerType::Int32, Box::new(DataType::Utf8), false),
                false,
            ))
        }
        AvroSchema::Fixed(Fixed { size, logical, .. }) => match logical {
            Some(logical) => match logical {
                avro_schema::schema::FixedLogical::Decimal(precision, scale) => {
                    DataType::Decimal(*precision, *scale)
                }
                avro_schema::schema::FixedLogical::Duration => {
                    DataType::Interval(IntervalUnit::MonthDayNano)
                }
            },
            None => DataType::FixedSizeBinary(*size),
        },
    };

    let name = name.unwrap_or_default();

    Ok(Field::new(name, data_type, nullable).with_metadata(props))
}
