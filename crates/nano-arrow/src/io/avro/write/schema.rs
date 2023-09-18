use avro_schema::schema::{
    BytesLogical, Field as AvroField, Fixed, FixedLogical, IntLogical, LongLogical, Record,
    Schema as AvroSchema,
};

use crate::datatypes::*;
use crate::error::{Error, Result};

/// Converts a [`Schema`] to an Avro [`Record`].
pub fn to_record(schema: &Schema) -> Result<Record> {
    let mut name_counter: i32 = 0;
    let fields = schema
        .fields
        .iter()
        .map(|f| field_to_field(f, &mut name_counter))
        .collect::<Result<_>>()?;
    Ok(Record {
        name: "".to_string(),
        namespace: None,
        doc: None,
        aliases: vec![],
        fields,
    })
}

fn field_to_field(field: &Field, name_counter: &mut i32) -> Result<AvroField> {
    let schema = type_to_schema(field.data_type(), field.is_nullable, name_counter)?;
    Ok(AvroField::new(&field.name, schema))
}

fn type_to_schema(
    data_type: &DataType,
    is_nullable: bool,
    name_counter: &mut i32,
) -> Result<AvroSchema> {
    Ok(if is_nullable {
        AvroSchema::Union(vec![
            AvroSchema::Null,
            _type_to_schema(data_type, name_counter)?,
        ])
    } else {
        _type_to_schema(data_type, name_counter)?
    })
}

fn _get_field_name(name_counter: &mut i32) -> String {
    *name_counter += 1;
    format!("r{name_counter}")
}

fn _type_to_schema(data_type: &DataType, name_counter: &mut i32) -> Result<AvroSchema> {
    Ok(match data_type.to_logical_type() {
        DataType::Null => AvroSchema::Null,
        DataType::Boolean => AvroSchema::Boolean,
        DataType::Int32 => AvroSchema::Int(None),
        DataType::Int64 => AvroSchema::Long(None),
        DataType::Float32 => AvroSchema::Float,
        DataType::Float64 => AvroSchema::Double,
        DataType::Binary => AvroSchema::Bytes(None),
        DataType::LargeBinary => AvroSchema::Bytes(None),
        DataType::Utf8 => AvroSchema::String(None),
        DataType::LargeUtf8 => AvroSchema::String(None),
        DataType::LargeList(inner) | DataType::List(inner) => AvroSchema::Array(Box::new(
            type_to_schema(&inner.data_type, inner.is_nullable, name_counter)?,
        )),
        DataType::Struct(fields) => AvroSchema::Record(Record::new(
            _get_field_name(name_counter),
            fields
                .iter()
                .map(|f| field_to_field(f, name_counter))
                .collect::<Result<Vec<_>>>()?,
        )),
        DataType::Date32 => AvroSchema::Int(Some(IntLogical::Date)),
        DataType::Time32(TimeUnit::Millisecond) => AvroSchema::Int(Some(IntLogical::Time)),
        DataType::Time64(TimeUnit::Microsecond) => AvroSchema::Long(Some(LongLogical::Time)),
        DataType::Timestamp(TimeUnit::Millisecond, None) => {
            AvroSchema::Long(Some(LongLogical::LocalTimestampMillis))
        }
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            AvroSchema::Long(Some(LongLogical::LocalTimestampMicros))
        }
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            let mut fixed = Fixed::new("", 12);
            fixed.logical = Some(FixedLogical::Duration);
            AvroSchema::Fixed(fixed)
        }
        DataType::FixedSizeBinary(size) => AvroSchema::Fixed(Fixed::new("", *size)),
        DataType::Decimal(p, s) => AvroSchema::Bytes(Some(BytesLogical::Decimal(*p, *s))),
        other => return Err(Error::NotYetImplemented(format!("write {other:?} to avro"))),
    })
}
