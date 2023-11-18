use avro_schema::schema::{Record, Schema as AvroSchema};
use avro_schema::write::encode;

use super::super::super::iterator::*;
use crate::array::*;
use crate::bitmap::utils::ZipValidity;
use crate::datatypes::{ArrowDataType, IntervalUnit, PhysicalType, PrimitiveType};
use crate::offset::Offset;
use crate::types::months_days_ns;

// Zigzag representation of false and true respectively.
const IS_NULL: u8 = 0;
const IS_VALID: u8 = 2;

/// A type alias for a boxed [`StreamingIterator`], used to write arrays into avro rows
/// (i.e. a column -> row transposition of types known at run-time)
pub type BoxSerializer<'a> = Box<dyn StreamingIterator<Item = [u8]> + 'a + Send + Sync>;

fn utf8_required<O: Offset>(array: &Utf8Array<O>) -> BoxSerializer {
    Box::new(BufStreamingIterator::new(
        array.values_iter(),
        |x, buf| {
            encode::zigzag_encode(x.len() as i64, buf).unwrap();
            buf.extend_from_slice(x.as_bytes());
        },
        vec![],
    ))
}

fn utf8_optional<O: Offset>(array: &Utf8Array<O>) -> BoxSerializer {
    Box::new(BufStreamingIterator::new(
        array.iter(),
        |x, buf| {
            if let Some(x) = x {
                buf.push(IS_VALID);
                encode::zigzag_encode(x.len() as i64, buf).unwrap();
                buf.extend_from_slice(x.as_bytes());
            } else {
                buf.push(IS_NULL);
            }
        },
        vec![],
    ))
}

fn binary_required<O: Offset>(array: &BinaryArray<O>) -> BoxSerializer {
    Box::new(BufStreamingIterator::new(
        array.values_iter(),
        |x, buf| {
            encode::zigzag_encode(x.len() as i64, buf).unwrap();
            buf.extend_from_slice(x);
        },
        vec![],
    ))
}

fn binary_optional<O: Offset>(array: &BinaryArray<O>) -> BoxSerializer {
    Box::new(BufStreamingIterator::new(
        array.iter(),
        |x, buf| {
            if let Some(x) = x {
                buf.push(IS_VALID);
                encode::zigzag_encode(x.len() as i64, buf).unwrap();
                buf.extend_from_slice(x);
            } else {
                buf.push(IS_NULL);
            }
        },
        vec![],
    ))
}

fn fixed_size_binary_required(array: &FixedSizeBinaryArray) -> BoxSerializer {
    Box::new(BufStreamingIterator::new(
        array.values_iter(),
        |x, buf| {
            buf.extend_from_slice(x);
        },
        vec![],
    ))
}

fn fixed_size_binary_optional(array: &FixedSizeBinaryArray) -> BoxSerializer {
    Box::new(BufStreamingIterator::new(
        array.iter(),
        |x, buf| {
            if let Some(x) = x {
                buf.push(IS_VALID);
                buf.extend_from_slice(x);
            } else {
                buf.push(IS_NULL);
            }
        },
        vec![],
    ))
}

fn list_required<'a, O: Offset>(array: &'a ListArray<O>, schema: &AvroSchema) -> BoxSerializer<'a> {
    let mut inner = new_serializer(array.values().as_ref(), schema);
    let lengths = array
        .offsets()
        .buffer()
        .windows(2)
        .map(|w| (w[1] - w[0]).to_usize() as i64);

    Box::new(BufStreamingIterator::new(
        lengths,
        move |length, buf| {
            encode::zigzag_encode(length, buf).unwrap();
            let mut rows = 0;
            while let Some(item) = inner.next() {
                buf.extend_from_slice(item);
                rows += 1;
                if rows == length {
                    encode::zigzag_encode(0, buf).unwrap();
                    break;
                }
            }
        },
        vec![],
    ))
}

fn list_optional<'a, O: Offset>(array: &'a ListArray<O>, schema: &AvroSchema) -> BoxSerializer<'a> {
    let mut inner = new_serializer(array.values().as_ref(), schema);
    let lengths = array
        .offsets()
        .buffer()
        .windows(2)
        .map(|w| (w[1] - w[0]).to_usize() as i64);
    let lengths = ZipValidity::new_with_validity(lengths, array.validity());

    Box::new(BufStreamingIterator::new(
        lengths,
        move |length, buf| {
            if let Some(length) = length {
                buf.push(IS_VALID);
                encode::zigzag_encode(length, buf).unwrap();
                let mut rows = 0;
                while let Some(item) = inner.next() {
                    buf.extend_from_slice(item);
                    rows += 1;
                    if rows == length {
                        encode::zigzag_encode(0, buf).unwrap();
                        break;
                    }
                }
            } else {
                buf.push(IS_NULL);
            }
        },
        vec![],
    ))
}

fn struct_required<'a>(array: &'a StructArray, schema: &Record) -> BoxSerializer<'a> {
    let schemas = schema.fields.iter().map(|x| &x.schema);
    let mut inner = array
        .values()
        .iter()
        .zip(schemas)
        .map(|(x, schema)| new_serializer(x.as_ref(), schema))
        .collect::<Vec<_>>();

    Box::new(BufStreamingIterator::new(
        0..array.len(),
        move |_, buf| {
            inner
                .iter_mut()
                .for_each(|item| buf.extend_from_slice(item.next().unwrap()))
        },
        vec![],
    ))
}

fn struct_optional<'a>(array: &'a StructArray, schema: &Record) -> BoxSerializer<'a> {
    let schemas = schema.fields.iter().map(|x| &x.schema);
    let mut inner = array
        .values()
        .iter()
        .zip(schemas)
        .map(|(x, schema)| new_serializer(x.as_ref(), schema))
        .collect::<Vec<_>>();

    let iterator = ZipValidity::new_with_validity(0..array.len(), array.validity());

    Box::new(BufStreamingIterator::new(
        iterator,
        move |maybe, buf| {
            if maybe.is_some() {
                buf.push(IS_VALID);
                inner
                    .iter_mut()
                    .for_each(|item| buf.extend_from_slice(item.next().unwrap()))
            } else {
                buf.push(IS_NULL);
                // skip the item
                inner.iter_mut().for_each(|item| {
                    let _ = item.next().unwrap();
                });
            }
        },
        vec![],
    ))
}

/// Creates a [`StreamingIterator`] trait object that presents items from `array`
/// encoded according to `schema`.
/// # Panic
/// This function panics iff the `data_type` is not supported (use [`can_serialize`] to check)
/// # Implementation
/// This function performs minimal CPU work: it dynamically dispatches based on the schema
/// and arrow type.
pub fn new_serializer<'a>(array: &'a dyn Array, schema: &AvroSchema) -> BoxSerializer<'a> {
    let data_type = array.data_type().to_physical_type();

    match (data_type, schema) {
        (PhysicalType::Boolean, AvroSchema::Boolean) => {
            let values = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            Box::new(BufStreamingIterator::new(
                values.values_iter(),
                |x, buf| {
                    buf.push(x as u8);
                },
                vec![],
            ))
        },
        (PhysicalType::Boolean, AvroSchema::Union(_)) => {
            let values = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            Box::new(BufStreamingIterator::new(
                values.iter(),
                |x, buf| {
                    if let Some(x) = x {
                        buf.extend_from_slice(&[IS_VALID, x as u8]);
                    } else {
                        buf.push(IS_NULL);
                    }
                },
                vec![],
            ))
        },
        (PhysicalType::Utf8, AvroSchema::Union(_)) => {
            utf8_optional::<i32>(array.as_any().downcast_ref().unwrap())
        },
        (PhysicalType::LargeUtf8, AvroSchema::Union(_)) => {
            utf8_optional::<i64>(array.as_any().downcast_ref().unwrap())
        },
        (PhysicalType::Utf8, AvroSchema::String(_)) => {
            utf8_required::<i32>(array.as_any().downcast_ref().unwrap())
        },
        (PhysicalType::LargeUtf8, AvroSchema::String(_)) => {
            utf8_required::<i64>(array.as_any().downcast_ref().unwrap())
        },
        (PhysicalType::Binary, AvroSchema::Union(_)) => {
            binary_optional::<i32>(array.as_any().downcast_ref().unwrap())
        },
        (PhysicalType::LargeBinary, AvroSchema::Union(_)) => {
            binary_optional::<i64>(array.as_any().downcast_ref().unwrap())
        },
        (PhysicalType::FixedSizeBinary, AvroSchema::Union(_)) => {
            fixed_size_binary_optional(array.as_any().downcast_ref().unwrap())
        },
        (PhysicalType::Binary, AvroSchema::Bytes(_)) => {
            binary_required::<i32>(array.as_any().downcast_ref().unwrap())
        },
        (PhysicalType::LargeBinary, AvroSchema::Bytes(_)) => {
            binary_required::<i64>(array.as_any().downcast_ref().unwrap())
        },
        (PhysicalType::FixedSizeBinary, AvroSchema::Fixed(_)) => {
            fixed_size_binary_required(array.as_any().downcast_ref().unwrap())
        },

        (PhysicalType::Primitive(PrimitiveType::Int32), AvroSchema::Union(_)) => {
            let values = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .unwrap();
            Box::new(BufStreamingIterator::new(
                values.iter(),
                |x, buf| {
                    if let Some(x) = x {
                        buf.push(IS_VALID);
                        encode::zigzag_encode(*x as i64, buf).unwrap();
                    } else {
                        buf.push(IS_NULL);
                    }
                },
                vec![],
            ))
        },
        (PhysicalType::Primitive(PrimitiveType::Int32), AvroSchema::Int(_)) => {
            let values = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .unwrap();
            Box::new(BufStreamingIterator::new(
                values.values().iter(),
                |x, buf| {
                    encode::zigzag_encode(*x as i64, buf).unwrap();
                },
                vec![],
            ))
        },
        (PhysicalType::Primitive(PrimitiveType::Int64), AvroSchema::Union(_)) => {
            let values = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .unwrap();
            Box::new(BufStreamingIterator::new(
                values.iter(),
                |x, buf| {
                    if let Some(x) = x {
                        buf.push(IS_VALID);
                        encode::zigzag_encode(*x, buf).unwrap();
                    } else {
                        buf.push(IS_NULL);
                    }
                },
                vec![],
            ))
        },
        (PhysicalType::Primitive(PrimitiveType::Int64), AvroSchema::Long(_)) => {
            let values = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .unwrap();
            Box::new(BufStreamingIterator::new(
                values.values().iter(),
                |x, buf| {
                    encode::zigzag_encode(*x, buf).unwrap();
                },
                vec![],
            ))
        },
        (PhysicalType::Primitive(PrimitiveType::Float32), AvroSchema::Union(_)) => {
            let values = array
                .as_any()
                .downcast_ref::<PrimitiveArray<f32>>()
                .unwrap();
            Box::new(BufStreamingIterator::new(
                values.iter(),
                |x, buf| {
                    if let Some(x) = x {
                        buf.push(IS_VALID);
                        buf.extend(x.to_le_bytes())
                    } else {
                        buf.push(IS_NULL);
                    }
                },
                vec![],
            ))
        },
        (PhysicalType::Primitive(PrimitiveType::Float32), AvroSchema::Float) => {
            let values = array
                .as_any()
                .downcast_ref::<PrimitiveArray<f32>>()
                .unwrap();
            Box::new(BufStreamingIterator::new(
                values.values().iter(),
                |x, buf| {
                    buf.extend_from_slice(&x.to_le_bytes());
                },
                vec![],
            ))
        },
        (PhysicalType::Primitive(PrimitiveType::Float64), AvroSchema::Union(_)) => {
            let values = array
                .as_any()
                .downcast_ref::<PrimitiveArray<f64>>()
                .unwrap();
            Box::new(BufStreamingIterator::new(
                values.iter(),
                |x, buf| {
                    if let Some(x) = x {
                        buf.push(IS_VALID);
                        buf.extend(x.to_le_bytes())
                    } else {
                        buf.push(IS_NULL);
                    }
                },
                vec![],
            ))
        },
        (PhysicalType::Primitive(PrimitiveType::Float64), AvroSchema::Double) => {
            let values = array
                .as_any()
                .downcast_ref::<PrimitiveArray<f64>>()
                .unwrap();
            Box::new(BufStreamingIterator::new(
                values.values().iter(),
                |x, buf| {
                    buf.extend_from_slice(&x.to_le_bytes());
                },
                vec![],
            ))
        },
        (PhysicalType::Primitive(PrimitiveType::Int128), AvroSchema::Bytes(_)) => {
            let values = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i128>>()
                .unwrap();
            Box::new(BufStreamingIterator::new(
                values.values().iter(),
                |x, buf| {
                    let len = ((x.leading_zeros() / 8) - ((x.leading_zeros() / 8) % 2)) as usize;
                    encode::zigzag_encode((16 - len) as i64, buf).unwrap();
                    buf.extend_from_slice(&x.to_be_bytes()[len..]);
                },
                vec![],
            ))
        },
        (PhysicalType::Primitive(PrimitiveType::Int128), AvroSchema::Union(_)) => {
            let values = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i128>>()
                .unwrap();
            Box::new(BufStreamingIterator::new(
                values.iter(),
                |x, buf| {
                    if let Some(x) = x {
                        buf.push(IS_VALID);
                        let len =
                            ((x.leading_zeros() / 8) - ((x.leading_zeros() / 8) % 2)) as usize;
                        encode::zigzag_encode((16 - len) as i64, buf).unwrap();
                        buf.extend_from_slice(&x.to_be_bytes()[len..]);
                    } else {
                        buf.push(IS_NULL);
                    }
                },
                vec![],
            ))
        },
        (PhysicalType::Primitive(PrimitiveType::MonthDayNano), AvroSchema::Fixed(_)) => {
            let values = array
                .as_any()
                .downcast_ref::<PrimitiveArray<months_days_ns>>()
                .unwrap();
            Box::new(BufStreamingIterator::new(
                values.values().iter(),
                interval_write,
                vec![],
            ))
        },
        (PhysicalType::Primitive(PrimitiveType::MonthDayNano), AvroSchema::Union(_)) => {
            let values = array
                .as_any()
                .downcast_ref::<PrimitiveArray<months_days_ns>>()
                .unwrap();
            Box::new(BufStreamingIterator::new(
                values.iter(),
                |x, buf| {
                    if let Some(x) = x {
                        buf.push(IS_VALID);
                        interval_write(x, buf)
                    } else {
                        buf.push(IS_NULL);
                    }
                },
                vec![],
            ))
        },

        (PhysicalType::List, AvroSchema::Array(schema)) => {
            list_required::<i32>(array.as_any().downcast_ref().unwrap(), schema.as_ref())
        },
        (PhysicalType::LargeList, AvroSchema::Array(schema)) => {
            list_required::<i64>(array.as_any().downcast_ref().unwrap(), schema.as_ref())
        },
        (PhysicalType::List, AvroSchema::Union(inner)) => {
            let schema = if let AvroSchema::Array(schema) = &inner[1] {
                schema.as_ref()
            } else {
                unreachable!("The schema declaration does not match the deserialization")
            };
            list_optional::<i32>(array.as_any().downcast_ref().unwrap(), schema)
        },
        (PhysicalType::LargeList, AvroSchema::Union(inner)) => {
            let schema = if let AvroSchema::Array(schema) = &inner[1] {
                schema.as_ref()
            } else {
                unreachable!("The schema declaration does not match the deserialization")
            };
            list_optional::<i64>(array.as_any().downcast_ref().unwrap(), schema)
        },
        (PhysicalType::Struct, AvroSchema::Record(inner)) => {
            struct_required(array.as_any().downcast_ref().unwrap(), inner)
        },
        (PhysicalType::Struct, AvroSchema::Union(inner)) => {
            let inner = if let AvroSchema::Record(inner) = &inner[1] {
                inner
            } else {
                unreachable!("The schema declaration does not match the deserialization")
            };
            struct_optional(array.as_any().downcast_ref().unwrap(), inner)
        },
        (a, b) => todo!("{:?} -> {:?} not supported", a, b),
    }
}

/// Whether [`new_serializer`] supports `data_type`.
pub fn can_serialize(data_type: &ArrowDataType) -> bool {
    use ArrowDataType::*;
    match data_type.to_logical_type() {
        List(inner) => return can_serialize(&inner.data_type),
        LargeList(inner) => return can_serialize(&inner.data_type),
        Struct(inner) => return inner.iter().all(|inner| can_serialize(&inner.data_type)),
        _ => {},
    };

    matches!(
        data_type,
        Boolean
            | Int32
            | Int64
            | Float32
            | Float64
            | Decimal(_, _)
            | Utf8
            | Binary
            | FixedSizeBinary(_)
            | LargeUtf8
            | LargeBinary
            | Interval(IntervalUnit::MonthDayNano)
    )
}

#[inline]
fn interval_write(x: &months_days_ns, buf: &mut Vec<u8>) {
    // https://avro.apache.org/docs/current/spec.html#Duration
    // 12 bytes, months, days, millis in LE
    buf.reserve(12);
    buf.extend(x.months().to_le_bytes());
    buf.extend(x.days().to_le_bytes());
    buf.extend(((x.ns() / 1_000_000) as i32).to_le_bytes());
}
