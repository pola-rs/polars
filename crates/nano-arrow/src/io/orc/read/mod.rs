//! APIs to read from [ORC format](https://orc.apache.org).
use std::io::Read;

use crate::array::{Array, BinaryArray, BooleanArray, Int64Array, PrimitiveArray, Utf8Array};
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::datatypes::{DataType, Field, Schema};
use crate::error::Error;
use crate::offset::{Offset, Offsets};
use crate::types::NativeType;

use orc_format::proto::stream::Kind;
use orc_format::proto::{Footer, Type};
use orc_format::read::decode;
use orc_format::read::Column;

/// Infers a [`Schema`] from the files' [`Footer`].
/// # Errors
/// This function errors if the type is not yet supported.
pub fn infer_schema(footer: &Footer) -> Result<Schema, Error> {
    let types = &footer.types;

    let dt = infer_dt(&footer.types[0], types)?;
    if let DataType::Struct(fields) = dt {
        Ok(fields.into())
    } else {
        Err(Error::ExternalFormat(
            "ORC root type must be a struct".to_string(),
        ))
    }
}

fn infer_dt(type_: &Type, types: &[Type]) -> Result<DataType, Error> {
    use orc_format::proto::r#type::Kind::*;
    let dt = match type_.kind() {
        Boolean => DataType::Boolean,
        Byte => DataType::Int8,
        Short => DataType::Int16,
        Int => DataType::Int32,
        Long => DataType::Int64,
        Float => DataType::Float32,
        Double => DataType::Float64,
        String => DataType::Utf8,
        Binary => DataType::Binary,
        Struct => {
            let sub_types = type_
                .subtypes
                .iter()
                .cloned()
                .zip(type_.field_names.iter())
                .map(|(i, name)| {
                    infer_dt(
                        types.get(i as usize).ok_or_else(|| {
                            Error::ExternalFormat(format!("ORC field {i} not found"))
                        })?,
                        types,
                    )
                    .map(|dt| Field::new(name, dt, true))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            DataType::Struct(sub_types)
        }
        kind => return Err(Error::nyi(format!("Reading {kind:?} from ORC"))),
    };
    Ok(dt)
}

fn deserialize_validity(column: &Column, scratch: &mut Vec<u8>) -> Result<Option<Bitmap>, Error> {
    let stream = column.get_stream(Kind::Present, std::mem::take(scratch))?;

    let mut stream = decode::BooleanIter::new(stream, column.number_of_rows());

    let mut validity = MutableBitmap::with_capacity(column.number_of_rows());
    for item in stream.by_ref() {
        validity.push(item?)
    }

    *scratch = std::mem::take(&mut stream.into_inner().into_inner());

    Ok(validity.into())
}

/// Deserializes column `column` from `stripe`, assumed to represent a f32
fn deserialize_float<T: NativeType + decode::Float>(
    data_type: DataType,
    column: &Column,
) -> Result<PrimitiveArray<T>, Error> {
    let mut scratch = vec![];
    let num_rows = column.number_of_rows();

    let validity = deserialize_validity(column, &mut scratch)?;

    let mut chunks = column.get_stream(Kind::Data, scratch)?;

    let mut values = Vec::with_capacity(num_rows);
    if let Some(validity) = &validity {
        let mut items =
            decode::FloatIter::<T, _>::new(&mut chunks, validity.len() - validity.unset_bits());

        for is_valid in validity {
            if is_valid {
                values.push(items.next().transpose()?.unwrap_or_default())
            } else {
                values.push(T::default())
            }
        }
    } else {
        let items = decode::FloatIter::<T, _>::new(&mut chunks, num_rows);
        for item in items {
            values.push(item?);
        }
    }

    PrimitiveArray::try_new(data_type, values.into(), validity)
}

/// Deserializes column `column` from `stripe`, assumed to represent a boolean array
fn deserialize_bool(data_type: DataType, column: &Column) -> Result<BooleanArray, Error> {
    let num_rows = column.number_of_rows();
    let mut scratch = vec![];

    let validity = deserialize_validity(column, &mut scratch)?;

    let mut chunks = column.get_stream(Kind::Data, std::mem::take(&mut scratch))?;

    let mut values = MutableBitmap::with_capacity(num_rows);
    if let Some(validity) = &validity {
        let mut items =
            decode::BooleanIter::new(&mut chunks, validity.len() - validity.unset_bits());

        for is_valid in validity {
            values.push(if is_valid {
                items.next().transpose()?.unwrap_or_default()
            } else {
                false
            });
        }
    } else {
        let valid_iter = decode::BooleanIter::new(&mut chunks, num_rows);
        for v in valid_iter {
            values.push(v?)
        }
    }

    BooleanArray::try_new(data_type, values.into(), validity)
}

/// Deserializes column `column` from `stripe`, assumed to represent a boolean array
fn deserialize_i64(data_type: DataType, column: &Column) -> Result<Int64Array, Error> {
    let num_rows = column.number_of_rows();
    let mut scratch = vec![];

    let validity = deserialize_validity(column, &mut scratch)?;

    let chunks = column.get_stream(Kind::Data, std::mem::take(&mut scratch))?;

    let mut values = Vec::with_capacity(num_rows);
    if let Some(validity) = &validity {
        let mut iter =
            decode::SignedRleV2Iter::new(chunks, validity.len() - validity.unset_bits(), vec![]);

        for is_valid in validity {
            if is_valid {
                let item = iter.next().transpose()?.unwrap_or_default();
                values.push(item);
            } else {
                values.push(0);
            }
        }
    } else {
        let mut iter = decode::SignedRleV2RunIter::new(chunks, num_rows, vec![]);
        iter.try_for_each(|run| {
            run.map(|run| match run {
                decode::SignedRleV2Run::Direct(values_iter) => values.extend(values_iter),
                decode::SignedRleV2Run::Delta(values_iter) => values.extend(values_iter),
                decode::SignedRleV2Run::ShortRepeat(values_iter) => values.extend(values_iter),
            })
        })?;
    }

    Int64Array::try_new(data_type, values.into(), validity)
}

/// Deserializes column `column` from `stripe`, assumed to represent a boolean array
fn deserialize_int<T>(data_type: DataType, column: &Column) -> Result<PrimitiveArray<T>, Error>
where
    T: NativeType + TryFrom<i64>,
{
    let num_rows = column.number_of_rows();
    let mut scratch = vec![];

    let validity = deserialize_validity(column, &mut scratch)?;

    let chunks = column.get_stream(Kind::Data, std::mem::take(&mut scratch))?;

    let mut values = Vec::<T>::with_capacity(num_rows);
    if let Some(validity) = &validity {
        let validity_iter = validity.iter();

        let mut iter =
            decode::SignedRleV2Iter::new(chunks, validity.len() - validity.unset_bits(), vec![]);
        for is_valid in validity_iter {
            if is_valid {
                let item = iter.next().transpose()?.unwrap_or_default();
                let item = item
                    .try_into()
                    .map_err(|_| Error::ExternalFormat("value uncastable".to_string()))?;
                values.push(item);
            } else {
                values.push(T::default());
            }
        }
    } else {
        let mut iter = decode::SignedRleV2RunIter::new(chunks, num_rows, vec![]);

        iter.try_for_each(|run| {
            run.map_err(Error::from).and_then(|run| match run {
                decode::SignedRleV2Run::Direct(values_iter) => {
                    for item in values_iter {
                        let item = item
                            .try_into()
                            .map_err(|_| Error::ExternalFormat("value uncastable".to_string()))?;
                        values.push(item);
                    }
                    Ok(())
                }
                decode::SignedRleV2Run::Delta(values_iter) => {
                    for item in values_iter {
                        let item = item
                            .try_into()
                            .map_err(|_| Error::ExternalFormat("value uncastable".to_string()))?;
                        values.push(item);
                    }
                    Ok(())
                }
                decode::SignedRleV2Run::ShortRepeat(values_iter) => {
                    for item in values_iter {
                        let item = item
                            .try_into()
                            .map_err(|_| Error::ExternalFormat("value uncastable".to_string()))?;
                        values.push(item);
                    }
                    Ok(())
                }
            })
        })?;
    }

    PrimitiveArray::try_new(data_type, values.into(), validity)
}

#[inline]
fn try_extend<O: Offset + TryFrom<u64>, I: Iterator<Item = u64>>(
    offsets: &mut Offsets<O>,
    iter: I,
) -> Result<(), Error> {
    for item in iter {
        let length: O = item.try_into().map_err(|_| Error::Overflow)?;
        offsets.try_push(length)?
    }
    Ok(())
}

fn deserialize_binary_generic<O: Offset + TryFrom<u64>>(
    column: &Column,
) -> Result<(Offsets<O>, Vec<u8>, Option<Bitmap>), Error> {
    let num_rows = column.number_of_rows();
    let mut scratch = vec![];

    let validity = deserialize_validity(column, &mut scratch)?;

    let lengths = column.get_stream(Kind::Length, scratch)?;

    let mut offsets = Offsets::with_capacity(num_rows);
    if let Some(validity) = &validity {
        let mut iter =
            decode::UnsignedRleV2Iter::new(lengths, validity.len() - validity.unset_bits(), vec![]);
        for is_valid in validity {
            if is_valid {
                let item = iter
                    .next()
                    .transpose()?
                    .ok_or(orc_format::error::Error::OutOfSpec)?;
                let length: O = item
                    .try_into()
                    .map_err(|_| Error::ExternalFormat("value uncastable".to_string()))?;
                offsets.try_push(length)?;
            } else {
                offsets.extend_constant(1)
            }
        }
        let (lengths, _) = iter.into_inner();
        scratch = std::mem::take(&mut lengths.into_inner());
    } else {
        let mut iter = decode::UnsignedRleV2RunIter::new(lengths, num_rows, vec![]);
        iter.try_for_each(|run| {
            run.map_err(Error::from).and_then(|run| match run {
                decode::UnsignedRleV2Run::Direct(values_iter) => {
                    try_extend(&mut offsets, values_iter)
                }
                decode::UnsignedRleV2Run::Delta(values_iter) => {
                    try_extend(&mut offsets, values_iter)
                }
                decode::UnsignedRleV2Run::ShortRepeat(values_iter) => {
                    try_extend(&mut offsets, values_iter)
                }
            })
        })?;
        let (lengths, _) = iter.into_inner();
        scratch = std::mem::take(&mut lengths.into_inner());
    }
    let length = offsets.last().to_usize();
    let mut values = vec![0; length];

    let mut data = column.get_stream(Kind::Data, scratch)?;
    data.read_exact(&mut values)?;

    Ok((offsets, values, validity))
}

fn deserialize_utf8<O: Offset + TryFrom<u64>>(
    data_type: DataType,
    column: &Column,
) -> Result<Utf8Array<O>, Error> {
    let (offsets, values, validity) = deserialize_binary_generic::<O>(column)?;
    Utf8Array::try_new(data_type, offsets.into(), values.into(), validity)
}

fn deserialize_binary<O: Offset + TryFrom<u64>>(
    data_type: DataType,
    column: &Column,
) -> Result<BinaryArray<O>, Error> {
    let (offsets, values, validity) = deserialize_binary_generic::<O>(column)?;
    BinaryArray::try_new(data_type, offsets.into(), values.into(), validity)
}

/// Deserializes column `column` from `stripe`, assumed
/// to represent an array of `data_type`.
pub fn deserialize(data_type: DataType, column: &Column) -> Result<Box<dyn Array>, Error> {
    match data_type {
        DataType::Boolean => deserialize_bool(data_type, column).map(|x| x.boxed()),
        DataType::Int8 => deserialize_int::<i8>(data_type, column).map(|x| x.boxed()),
        DataType::Int16 => deserialize_int::<i16>(data_type, column).map(|x| x.boxed()),
        DataType::Int32 => deserialize_int::<i32>(data_type, column).map(|x| x.boxed()),
        DataType::Int64 => deserialize_i64(data_type, column).map(|x| x.boxed()),
        DataType::Float32 => deserialize_float::<f32>(data_type, column).map(|x| x.boxed()),
        DataType::Float64 => deserialize_float::<f64>(data_type, column).map(|x| x.boxed()),
        DataType::Utf8 => deserialize_utf8::<i32>(data_type, column).map(|x| x.boxed()),
        DataType::LargeUtf8 => deserialize_utf8::<i64>(data_type, column).map(|x| x.boxed()),
        DataType::Binary => deserialize_binary::<i32>(data_type, column).map(|x| x.boxed()),
        DataType::LargeBinary => deserialize_binary::<i64>(data_type, column).map(|x| x.boxed()),
        dt => Err(Error::nyi(format!("Deserializing {dt:?} from ORC"))),
    }
}
