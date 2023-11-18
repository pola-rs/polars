use avro_schema::schema::{
    BytesLogical, Field as AvroField, Fixed, FixedLogical, IntLogical, LongLogical, Record,
    Schema as AvroSchema,
};
use polars_error::{polars_bail, PolarsResult};

use crate::datatypes::*;

/// Converts a [`ArrowSchema`] to an Avro [`Record`].
pub fn to_record(schema: &ArrowSchema, name: String) -> PolarsResult<Record> {
    let mut name_counter: i32 = 0;
    let fields = schema
        .fields
        .iter()
        .map(|f| field_to_field(f, &mut name_counter))
        .collect::<PolarsResult<_>>()?;
    Ok(Record {
        name,
        namespace: None,
        doc: None,
        aliases: vec![],
        fields,
    })
}

fn field_to_field(field: &Field, name_counter: &mut i32) -> PolarsResult<AvroField> {
    let schema = type_to_schema(field.data_type(), field.is_nullable, name_counter)?;
    Ok(AvroField::new(&field.name, schema))
}

fn type_to_schema(
    data_type: &ArrowDataType,
    is_nullable: bool,
    name_counter: &mut i32,
) -> PolarsResult<AvroSchema> {
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

fn _type_to_schema(data_type: &ArrowDataType, name_counter: &mut i32) -> PolarsResult<AvroSchema> {
    Ok(match data_type.to_logical_type() {
        ArrowDataType::Null => AvroSchema::Null,
        ArrowDataType::Boolean => AvroSchema::Boolean,
        ArrowDataType::Int32 => AvroSchema::Int(None),
        ArrowDataType::Int64 => AvroSchema::Long(None),
        ArrowDataType::Float32 => AvroSchema::Float,
        ArrowDataType::Float64 => AvroSchema::Double,
        ArrowDataType::Binary => AvroSchema::Bytes(None),
        ArrowDataType::LargeBinary => AvroSchema::Bytes(None),
        ArrowDataType::Utf8 => AvroSchema::String(None),
        ArrowDataType::LargeUtf8 => AvroSchema::String(None),
        ArrowDataType::LargeList(inner) | ArrowDataType::List(inner) => {
            AvroSchema::Array(Box::new(type_to_schema(
                &inner.data_type,
                inner.is_nullable,
                name_counter,
            )?))
        },
        ArrowDataType::Struct(fields) => AvroSchema::Record(Record::new(
            _get_field_name(name_counter),
            fields
                .iter()
                .map(|f| field_to_field(f, name_counter))
                .collect::<PolarsResult<Vec<_>>>()?,
        )),
        ArrowDataType::Date32 => AvroSchema::Int(Some(IntLogical::Date)),
        ArrowDataType::Time32(TimeUnit::Millisecond) => AvroSchema::Int(Some(IntLogical::Time)),
        ArrowDataType::Time64(TimeUnit::Microsecond) => AvroSchema::Long(Some(LongLogical::Time)),
        ArrowDataType::Timestamp(TimeUnit::Millisecond, None) => {
            AvroSchema::Long(Some(LongLogical::LocalTimestampMillis))
        },
        ArrowDataType::Timestamp(TimeUnit::Microsecond, None) => {
            AvroSchema::Long(Some(LongLogical::LocalTimestampMicros))
        },
        ArrowDataType::Interval(IntervalUnit::MonthDayNano) => {
            let mut fixed = Fixed::new("", 12);
            fixed.logical = Some(FixedLogical::Duration);
            AvroSchema::Fixed(fixed)
        },
        ArrowDataType::FixedSizeBinary(size) => AvroSchema::Fixed(Fixed::new("", *size)),
        ArrowDataType::Decimal(p, s) => AvroSchema::Bytes(Some(BytesLogical::Decimal(*p, *s))),
        other => polars_bail!(nyi = "write {other:?} to avro"),
    })
}
