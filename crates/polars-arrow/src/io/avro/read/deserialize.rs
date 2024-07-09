use avro_schema::file::Block;
use avro_schema::schema::{Enum, Field as AvroField, Record, Schema as AvroSchema};
use polars_error::{polars_bail, polars_err, PolarsResult};

use super::nested::*;
use super::util;
use crate::array::*;
use crate::datatypes::*;
use crate::record_batch::RecordBatchT;
use crate::types::months_days_ns;
use crate::with_match_primitive_type_full;

fn make_mutable(
    data_type: &ArrowDataType,
    avro_field: Option<&AvroSchema>,
    capacity: usize,
) -> PolarsResult<Box<dyn MutableArray>> {
    Ok(match data_type.to_physical_type() {
        PhysicalType::Boolean => {
            Box::new(MutableBooleanArray::with_capacity(capacity)) as Box<dyn MutableArray>
        },
        PhysicalType::Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            Box::new(MutablePrimitiveArray::<$T>::with_capacity(capacity).to(data_type.clone()))
                as Box<dyn MutableArray>
        }),
        PhysicalType::Binary => {
            Box::new(MutableBinaryArray::<i32>::with_capacity(capacity)) as Box<dyn MutableArray>
        },
        PhysicalType::Utf8 => {
            Box::new(MutableUtf8Array::<i32>::with_capacity(capacity)) as Box<dyn MutableArray>
        },
        PhysicalType::Dictionary(_) => {
            if let Some(AvroSchema::Enum(Enum { symbols, .. })) = avro_field {
                let values = Utf8Array::<i32>::from_slice(symbols);
                Box::new(FixedItemsUtf8Dictionary::with_capacity(values, capacity))
                    as Box<dyn MutableArray>
            } else {
                unreachable!()
            }
        },
        _ => match data_type {
            ArrowDataType::List(inner) => {
                let values = make_mutable(inner.data_type(), None, 0)?;
                Box::new(DynMutableListArray::<i32>::new_from(
                    values,
                    data_type.clone(),
                    capacity,
                )) as Box<dyn MutableArray>
            },
            ArrowDataType::FixedSizeBinary(size) => {
                Box::new(MutableFixedSizeBinaryArray::with_capacity(*size, capacity))
                    as Box<dyn MutableArray>
            },
            ArrowDataType::Struct(fields) => {
                let values = fields
                    .iter()
                    .map(|field| make_mutable(field.data_type(), None, capacity))
                    .collect::<PolarsResult<Vec<_>>>()?;
                Box::new(DynMutableStructArray::new(values, data_type.clone()))
                    as Box<dyn MutableArray>
            },
            other => {
                polars_bail!(nyi = "Deserializing type {other:#?} is still not implemented")
            },
        },
    })
}

fn is_union_null_first(avro_field: &AvroSchema) -> bool {
    if let AvroSchema::Union(schemas) = avro_field {
        schemas[0] == AvroSchema::Null
    } else {
        unreachable!()
    }
}

fn deserialize_item<'a>(
    array: &mut dyn MutableArray,
    is_nullable: bool,
    avro_field: &AvroSchema,
    mut block: &'a [u8],
) -> PolarsResult<&'a [u8]> {
    if is_nullable {
        let variant = util::zigzag_i64(&mut block)?;
        let is_null_first = is_union_null_first(avro_field);
        if is_null_first && variant == 0 || !is_null_first && variant != 0 {
            array.push_null();
            return Ok(block);
        }
    }
    deserialize_value(array, avro_field, block)
}

fn deserialize_value<'a>(
    array: &mut dyn MutableArray,
    avro_field: &AvroSchema,
    mut block: &'a [u8],
) -> PolarsResult<&'a [u8]> {
    let data_type = array.data_type();
    match data_type {
        ArrowDataType::List(inner) => {
            let is_nullable = inner.is_nullable;
            let avro_inner = match avro_field {
                AvroSchema::Array(inner) => inner.as_ref(),
                AvroSchema::Union(u) => match &u.as_slice() {
                    &[AvroSchema::Array(inner), _] | &[_, AvroSchema::Array(inner)] => {
                        inner.as_ref()
                    },
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };

            let array = array
                .as_mut_any()
                .downcast_mut::<DynMutableListArray<i32>>()
                .unwrap();
            // Arrays are encoded as a series of blocks.
            loop {
                // Each block consists of a long count value, followed by that many array items.
                let len = util::zigzag_i64(&mut block)?;
                let len = if len < 0 {
                    // Avro spec: If a block's count is negative, its absolute value is used,
                    // and the count is followed immediately by a long block size indicating the number of bytes in the block. This block size permits fast skipping through data, e.g., when projecting a record to a subset of its fields.
                    let _ = util::zigzag_i64(&mut block)?;

                    -len
                } else {
                    len
                };

                // A block with count zero indicates the end of the array.
                if len == 0 {
                    break;
                }

                // Each item is encoded per the array’s item schema.
                let values = array.mut_values();
                for _ in 0..len {
                    block = deserialize_item(values, is_nullable, avro_inner, block)?;
                }
            }
            array.try_push_valid()?;
        },
        ArrowDataType::Struct(inner_fields) => {
            let fields = match avro_field {
                AvroSchema::Record(Record { fields, .. }) => fields,
                AvroSchema::Union(u) => match &u.as_slice() {
                    &[AvroSchema::Record(Record { fields, .. }), _]
                    | &[_, AvroSchema::Record(Record { fields, .. })] => fields,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };

            let is_nullable = inner_fields
                .iter()
                .map(|x| x.is_nullable)
                .collect::<Vec<_>>();
            let array = array
                .as_mut_any()
                .downcast_mut::<DynMutableStructArray>()
                .unwrap();

            for (index, (field, is_nullable)) in fields.iter().zip(is_nullable.iter()).enumerate() {
                let values = array.mut_values(index);
                block = deserialize_item(values, *is_nullable, &field.schema, block)?;
            }
            array.try_push_valid()?;
        },
        _ => match data_type.to_physical_type() {
            PhysicalType::Boolean => {
                let is_valid = block[0] == 1;
                block = &block[1..];
                let array = array
                    .as_mut_any()
                    .downcast_mut::<MutableBooleanArray>()
                    .unwrap();
                array.push(Some(is_valid))
            },
            PhysicalType::Primitive(primitive) => match primitive {
                PrimitiveType::Int32 => {
                    let value = util::zigzag_i64(&mut block)? as i32;
                    let array = array
                        .as_mut_any()
                        .downcast_mut::<MutablePrimitiveArray<i32>>()
                        .unwrap();
                    array.push(Some(value))
                },
                PrimitiveType::Int64 => {
                    let value = util::zigzag_i64(&mut block)?;
                    let array = array
                        .as_mut_any()
                        .downcast_mut::<MutablePrimitiveArray<i64>>()
                        .unwrap();
                    array.push(Some(value))
                },
                PrimitiveType::Float32 => {
                    let value =
                        f32::from_le_bytes(block[..std::mem::size_of::<f32>()].try_into().unwrap());
                    block = &block[std::mem::size_of::<f32>()..];
                    let array = array
                        .as_mut_any()
                        .downcast_mut::<MutablePrimitiveArray<f32>>()
                        .unwrap();
                    array.push(Some(value))
                },
                PrimitiveType::Float64 => {
                    let value =
                        f64::from_le_bytes(block[..std::mem::size_of::<f64>()].try_into().unwrap());
                    block = &block[std::mem::size_of::<f64>()..];
                    let array = array
                        .as_mut_any()
                        .downcast_mut::<MutablePrimitiveArray<f64>>()
                        .unwrap();
                    array.push(Some(value))
                },
                PrimitiveType::MonthDayNano => {
                    // https://avro.apache.org/docs/current/spec.html#Duration
                    // 12 bytes, months, days, millis in LE
                    let data = &block[..12];
                    block = &block[12..];

                    let value = months_days_ns::new(
                        i32::from_le_bytes([data[0], data[1], data[2], data[3]]),
                        i32::from_le_bytes([data[4], data[5], data[6], data[7]]),
                        i32::from_le_bytes([data[8], data[9], data[10], data[11]]) as i64
                            * 1_000_000,
                    );

                    let array = array
                        .as_mut_any()
                        .downcast_mut::<MutablePrimitiveArray<months_days_ns>>()
                        .unwrap();
                    array.push(Some(value))
                },
                PrimitiveType::Int128 => {
                    let avro_inner = match avro_field {
                        AvroSchema::Bytes(_) | AvroSchema::Fixed(_) => avro_field,
                        AvroSchema::Union(u) => match &u.as_slice() {
                            &[e, AvroSchema::Null] | &[AvroSchema::Null, e] => e,
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    };
                    let len = match avro_inner {
                        AvroSchema::Bytes(_) => {
                            util::zigzag_i64(&mut block)?.try_into().map_err(|_| {
                                polars_err!(
                                    oos = "Avro format contains a non-usize number of bytes"
                                )
                            })?
                        },
                        AvroSchema::Fixed(b) => b.size,
                        _ => unreachable!(),
                    };
                    if len > 16 {
                        polars_bail!(oos = "Avro decimal bytes return more than 16 bytes")
                    }
                    let mut bytes = [0u8; 16];
                    bytes[..len].copy_from_slice(&block[..len]);
                    block = &block[len..];
                    let data = i128::from_be_bytes(bytes) >> (8 * (16 - len));
                    let array = array
                        .as_mut_any()
                        .downcast_mut::<MutablePrimitiveArray<i128>>()
                        .unwrap();
                    array.push(Some(data))
                },
                _ => unreachable!(),
            },
            PhysicalType::Utf8 => {
                let len: usize = util::zigzag_i64(&mut block)?.try_into().map_err(|_| {
                    polars_err!(oos = "Avro format contains a non-usize number of bytes")
                })?;
                let data = simdutf8::basic::from_utf8(&block[..len])?;
                block = &block[len..];

                let array = array
                    .as_mut_any()
                    .downcast_mut::<MutableUtf8Array<i32>>()
                    .unwrap();
                array.push(Some(data))
            },
            PhysicalType::Binary => {
                let len: usize = util::zigzag_i64(&mut block)?.try_into().map_err(|_| {
                    polars_err!(oos = "Avro format contains a non-usize number of bytes")
                })?;
                let data = &block[..len];
                block = &block[len..];

                let array = array
                    .as_mut_any()
                    .downcast_mut::<MutableBinaryArray<i32>>()
                    .unwrap();
                array.push(Some(data));
            },
            PhysicalType::FixedSizeBinary => {
                let array = array
                    .as_mut_any()
                    .downcast_mut::<MutableFixedSizeBinaryArray>()
                    .unwrap();
                let len = array.size();
                let data = &block[..len];
                block = &block[len..];
                array.push(Some(data));
            },
            PhysicalType::Dictionary(_) => {
                let index = util::zigzag_i64(&mut block)? as i32;
                let array = array
                    .as_mut_any()
                    .downcast_mut::<FixedItemsUtf8Dictionary>()
                    .unwrap();
                array.push_valid(index);
            },
            _ => todo!(),
        },
    };
    Ok(block)
}

fn skip_item<'a>(
    field: &Field,
    avro_field: &AvroSchema,
    mut block: &'a [u8],
) -> PolarsResult<&'a [u8]> {
    if field.is_nullable {
        let variant = util::zigzag_i64(&mut block)?;
        let is_null_first = is_union_null_first(avro_field);
        if is_null_first && variant == 0 || !is_null_first && variant != 0 {
            return Ok(block);
        }
    }
    match &field.data_type {
        ArrowDataType::List(inner) => {
            let avro_inner = match avro_field {
                AvroSchema::Array(inner) => inner.as_ref(),
                AvroSchema::Union(u) => match &u.as_slice() {
                    &[AvroSchema::Array(inner), _] | &[_, AvroSchema::Array(inner)] => {
                        inner.as_ref()
                    },
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };

            loop {
                let len = util::zigzag_i64(&mut block)?;
                let (len, bytes) = if len < 0 {
                    // Avro spec: If a block's count is negative, its absolute value is used,
                    // and the count is followed immediately by a long block size indicating the number of bytes in the block. This block size permits fast skipping through data, e.g., when projecting a record to a subset of its fields.
                    let bytes = util::zigzag_i64(&mut block)?;

                    (-len, Some(bytes))
                } else {
                    (len, None)
                };

                let bytes: Option<usize> = bytes
                    .map(|bytes| {
                        bytes
                            .try_into()
                            .map_err(|_| polars_err!(oos = "Avro block size negative or too large"))
                    })
                    .transpose()?;

                if len == 0 {
                    break;
                }

                if let Some(bytes) = bytes {
                    block = &block[bytes..];
                } else {
                    for _ in 0..len {
                        block = skip_item(inner, avro_inner, block)?;
                    }
                }
            }
        },
        ArrowDataType::Struct(inner_fields) => {
            let fields = match avro_field {
                AvroSchema::Record(Record { fields, .. }) => fields,
                AvroSchema::Union(u) => match &u.as_slice() {
                    &[AvroSchema::Record(Record { fields, .. }), _]
                    | &[_, AvroSchema::Record(Record { fields, .. })] => fields,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };

            for (field, avro_field) in inner_fields.iter().zip(fields.iter()) {
                block = skip_item(field, &avro_field.schema, block)?;
            }
        },
        _ => match field.data_type.to_physical_type() {
            PhysicalType::Boolean => {
                let _ = block[0] == 1;
                block = &block[1..];
            },
            PhysicalType::Primitive(primitive) => match primitive {
                PrimitiveType::Int32 => {
                    let _ = util::zigzag_i64(&mut block)?;
                },
                PrimitiveType::Int64 => {
                    let _ = util::zigzag_i64(&mut block)?;
                },
                PrimitiveType::Float32 => {
                    block = &block[std::mem::size_of::<f32>()..];
                },
                PrimitiveType::Float64 => {
                    block = &block[std::mem::size_of::<f64>()..];
                },
                PrimitiveType::MonthDayNano => {
                    block = &block[12..];
                },
                PrimitiveType::Int128 => {
                    let avro_inner = match avro_field {
                        AvroSchema::Bytes(_) | AvroSchema::Fixed(_) => avro_field,
                        AvroSchema::Union(u) => match &u.as_slice() {
                            &[e, AvroSchema::Null] | &[AvroSchema::Null, e] => e,
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    };
                    let len = match avro_inner {
                        AvroSchema::Bytes(_) => {
                            util::zigzag_i64(&mut block)?.try_into().map_err(|_| {
                                polars_err!(
                                    oos = "Avro format contains a non-usize number of bytes"
                                )
                            })?
                        },
                        AvroSchema::Fixed(b) => b.size,
                        _ => unreachable!(),
                    };
                    block = &block[len..];
                },
                _ => unreachable!(),
            },
            PhysicalType::Utf8 | PhysicalType::Binary => {
                let len: usize = util::zigzag_i64(&mut block)?.try_into().map_err(|_| {
                    polars_err!(oos = "Avro format contains a non-usize number of bytes")
                })?;
                block = &block[len..];
            },
            PhysicalType::FixedSizeBinary => {
                let len = if let ArrowDataType::FixedSizeBinary(len) = &field.data_type {
                    *len
                } else {
                    unreachable!()
                };

                block = &block[len..];
            },
            PhysicalType::Dictionary(_) => {
                let _ = util::zigzag_i64(&mut block)? as i32;
            },
            _ => todo!(),
        },
    }
    Ok(block)
}

/// Deserializes a [`Block`] assumed to be encoded according to [`AvroField`] into [`RecordBatchT`],
/// using `projection` to ignore `avro_fields`.
/// # Panics
/// `fields`, `avro_fields` and `projection` must have the same length.
pub fn deserialize(
    block: &Block,
    fields: &[Field],
    avro_fields: &[AvroField],
    projection: &[bool],
) -> PolarsResult<RecordBatchT<Box<dyn Array>>> {
    assert_eq!(fields.len(), avro_fields.len());
    assert_eq!(fields.len(), projection.len());

    let rows = block.number_of_rows;
    let mut block = block.data.as_ref();

    // create mutables, one per field
    let mut arrays: Vec<Box<dyn MutableArray>> = fields
        .iter()
        .zip(avro_fields.iter())
        .zip(projection.iter())
        .map(|((field, avro_field), projection)| {
            if *projection {
                make_mutable(&field.data_type, Some(&avro_field.schema), rows)
            } else {
                // just something; we are not going to use it
                make_mutable(&ArrowDataType::Int32, None, 0)
            }
        })
        .collect::<PolarsResult<_>>()?;

    // this is _the_ expensive transpose (rows -> columns)
    for _ in 0..rows {
        let iter = arrays
            .iter_mut()
            .zip(fields.iter())
            .zip(avro_fields.iter())
            .zip(projection.iter());

        for (((array, field), avro_field), projection) in iter {
            block = if *projection {
                deserialize_item(array.as_mut(), field.is_nullable, &avro_field.schema, block)
            } else {
                skip_item(field, &avro_field.schema, block)
            }?
        }
    }
    RecordBatchT::try_new(
        arrays
            .iter_mut()
            .zip(projection.iter())
            .filter_map(|x| x.1.then(|| x.0))
            .map(|array| array.as_box())
            .collect(),
    )
}
