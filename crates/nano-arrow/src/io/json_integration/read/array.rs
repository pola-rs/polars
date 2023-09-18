use ahash::AHashMap;
use num_traits::NumCast;
use serde_json::Value;

use crate::{
    array::*,
    bitmap::{Bitmap, MutableBitmap},
    buffer::Buffer,
    chunk::Chunk,
    datatypes::{DataType, PhysicalType, PrimitiveType, Schema},
    error::{Error, Result},
    io::ipc::IpcField,
    offset::Offset,
    types::{days_ms, i256, months_days_ns, NativeType},
};

use super::super::{ArrowJsonBatch, ArrowJsonColumn, ArrowJsonDictionaryBatch};

fn to_validity(validity: &Option<Vec<u8>>) -> Option<Bitmap> {
    validity.as_ref().and_then(|x| {
        x.iter()
            .map(|is_valid| *is_valid == 1)
            .collect::<MutableBitmap>()
            .into()
    })
}

fn to_offsets<O: Offset>(offsets: Option<&Vec<Value>>) -> Buffer<O> {
    offsets
        .as_ref()
        .unwrap()
        .iter()
        .map(|v| {
            match v {
                Value::String(s) => s.parse::<i64>().ok(),
                _ => v.as_i64(),
            }
            .map(|x| x as usize)
            .and_then(O::from_usize)
            .unwrap()
        })
        .collect()
}

fn to_days_ms(value: &Value) -> days_ms {
    if let Value::Object(v) = value {
        let days = v.get("days").unwrap();
        let milliseconds = v.get("milliseconds").unwrap();
        match (days, milliseconds) {
            (Value::Number(days), Value::Number(milliseconds)) => {
                let days = days.as_i64().unwrap() as i32;
                let milliseconds = milliseconds.as_i64().unwrap() as i32;
                days_ms::new(days, milliseconds)
            }
            (_, _) => panic!(),
        }
    } else {
        panic!()
    }
}

fn to_months_days_ns(value: &Value) -> months_days_ns {
    if let Value::Object(v) = value {
        let months = v.get("months").unwrap();
        let days = v.get("days").unwrap();
        let nanoseconds = v.get("nanoseconds").unwrap();
        match (months, days, nanoseconds) {
            (Value::Number(months), Value::Number(days), Value::Number(nanoseconds)) => {
                let months = months.as_i64().unwrap() as i32;
                let days = days.as_i64().unwrap() as i32;
                let nanoseconds = nanoseconds.as_i64().unwrap();
                months_days_ns::new(months, days, nanoseconds)
            }
            (_, _, _) => panic!(),
        }
    } else {
        panic!()
    }
}

fn to_primitive_days_ms(
    json_col: &ArrowJsonColumn,
    data_type: DataType,
) -> PrimitiveArray<days_ms> {
    let validity = to_validity(&json_col.validity);
    let values = json_col
        .data
        .as_ref()
        .unwrap()
        .iter()
        .map(to_days_ms)
        .collect();
    PrimitiveArray::<days_ms>::new(data_type, values, validity)
}

fn to_primitive_months_days_ns(
    json_col: &ArrowJsonColumn,
    data_type: DataType,
) -> PrimitiveArray<months_days_ns> {
    let validity = to_validity(&json_col.validity);
    let values = json_col
        .data
        .as_ref()
        .unwrap()
        .iter()
        .map(to_months_days_ns)
        .collect();
    PrimitiveArray::<months_days_ns>::new(data_type, values, validity)
}

fn to_decimal(json_col: &ArrowJsonColumn, data_type: DataType) -> PrimitiveArray<i128> {
    let validity = to_validity(&json_col.validity);
    let values = json_col
        .data
        .as_ref()
        .unwrap()
        .iter()
        .map(|value| match value {
            Value::String(x) => x.parse::<i128>().unwrap(),
            _ => {
                panic!()
            }
        })
        .collect();

    PrimitiveArray::<i128>::new(data_type, values, validity)
}

fn to_decimal256(json_col: &ArrowJsonColumn, data_type: DataType) -> PrimitiveArray<i256> {
    let validity = to_validity(&json_col.validity);
    let values = json_col
        .data
        .as_ref()
        .unwrap()
        .iter()
        .map(|value| match value {
            Value::String(x) => i256(x.parse::<ethnum::I256>().unwrap()),
            _ => {
                panic!()
            }
        })
        .collect();

    PrimitiveArray::<i256>::new(data_type, values, validity)
}

fn to_primitive<T: NativeType + NumCast>(
    json_col: &ArrowJsonColumn,
    data_type: DataType,
) -> PrimitiveArray<T> {
    let validity = to_validity(&json_col.validity);
    let values = if data_type == DataType::Float64 || data_type == DataType::Float32 {
        json_col
            .data
            .as_ref()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().and_then(num_traits::cast::<f64, T>).unwrap())
            .collect()
    } else {
        json_col
            .data
            .as_ref()
            .unwrap()
            .iter()
            .map(|value| match value {
                Value::Number(x) => x.as_i64().and_then(num_traits::cast::<i64, T>).unwrap(),
                Value::String(x) => x
                    .parse::<i64>()
                    .ok()
                    .and_then(num_traits::cast::<i64, T>)
                    .unwrap(),
                _ => {
                    panic!()
                }
            })
            .collect()
    };

    PrimitiveArray::<T>::new(data_type, values, validity)
}

fn to_binary<O: Offset>(json_col: &ArrowJsonColumn, data_type: DataType) -> Box<dyn Array> {
    let validity = to_validity(&json_col.validity);
    let offsets = to_offsets::<O>(json_col.offset.as_ref());
    let values = json_col
        .data
        .as_ref()
        .unwrap()
        .iter()
        .flat_map(|value| value.as_str().map(|x| hex::decode(x).unwrap()).unwrap())
        .collect();
    BinaryArray::new(data_type, offsets.try_into().unwrap(), values, validity).boxed()
}

fn to_utf8<O: Offset>(json_col: &ArrowJsonColumn, data_type: DataType) -> Box<dyn Array> {
    let validity = to_validity(&json_col.validity);
    let offsets = to_offsets::<O>(json_col.offset.as_ref());
    let values = json_col
        .data
        .as_ref()
        .unwrap()
        .iter()
        .flat_map(|value| value.as_str().unwrap().as_bytes().to_vec())
        .collect();
    Utf8Array::new(data_type, offsets.try_into().unwrap(), values, validity).boxed()
}

fn to_list<O: Offset>(
    json_col: &ArrowJsonColumn,
    data_type: DataType,
    field: &IpcField,
    dictionaries: &AHashMap<i64, ArrowJsonDictionaryBatch>,
) -> Result<Box<dyn Array>> {
    let validity = to_validity(&json_col.validity);

    let child_field = ListArray::<O>::get_child_field(&data_type);
    let children = &json_col.children.as_ref().unwrap()[0];
    let values = to_array(
        child_field.data_type().clone(),
        &field.fields[0],
        children,
        dictionaries,
    )?;
    let offsets = to_offsets::<O>(json_col.offset.as_ref());
    Ok(ListArray::<O>::new(data_type, offsets.try_into()?, values, validity).boxed())
}

fn to_map(
    json_col: &ArrowJsonColumn,
    data_type: DataType,
    field: &IpcField,
    dictionaries: &AHashMap<i64, ArrowJsonDictionaryBatch>,
) -> Result<Box<dyn Array>> {
    let validity = to_validity(&json_col.validity);

    let child_field = MapArray::get_field(&data_type);
    let children = &json_col.children.as_ref().unwrap()[0];
    let field = to_array(
        child_field.data_type().clone(),
        &field.fields[0],
        children,
        dictionaries,
    )?;
    let offsets = to_offsets::<i32>(json_col.offset.as_ref());
    Ok(Box::new(MapArray::new(
        data_type,
        offsets.try_into().unwrap(),
        field,
        validity,
    )))
}

fn to_dictionary<K: DictionaryKey + NumCast>(
    data_type: DataType,
    field: &IpcField,
    json_col: &ArrowJsonColumn,
    dictionaries: &AHashMap<i64, ArrowJsonDictionaryBatch>,
) -> Result<Box<dyn Array>> {
    // find dictionary
    let dict_id = field.dictionary_id.unwrap();
    let dictionary = dictionaries
        .get(&dict_id)
        .ok_or_else(|| Error::OutOfSpec(format!("Unable to find any dictionary id {dict_id}")))?;

    let keys = to_primitive(json_col, K::PRIMITIVE.into());

    let inner_data_type = DictionaryArray::<K>::try_get_child(&data_type)?;
    let values = to_array(
        inner_data_type.clone(),
        field,
        &dictionary.data.columns[0],
        dictionaries,
    )?;

    DictionaryArray::<K>::try_new(data_type, keys, values).map(|a| a.boxed())
}

/// Construct an [`Array`] from the JSON integration format
pub fn to_array(
    data_type: DataType,
    field: &IpcField,
    json_col: &ArrowJsonColumn,
    dictionaries: &AHashMap<i64, ArrowJsonDictionaryBatch>,
) -> Result<Box<dyn Array>> {
    use PhysicalType::*;
    match data_type.to_physical_type() {
        Null => Ok(Box::new(NullArray::new(data_type, json_col.count))),
        Boolean => {
            let validity = to_validity(&json_col.validity);
            let values = json_col
                .data
                .as_ref()
                .unwrap()
                .iter()
                .map(|value| value.as_bool().unwrap())
                .collect::<Bitmap>();
            Ok(Box::new(BooleanArray::new(data_type, values, validity)))
        }
        Primitive(PrimitiveType::Int8) => Ok(Box::new(to_primitive::<i8>(json_col, data_type))),
        Primitive(PrimitiveType::Int16) => Ok(Box::new(to_primitive::<i16>(json_col, data_type))),
        Primitive(PrimitiveType::Int32) => Ok(Box::new(to_primitive::<i32>(json_col, data_type))),
        Primitive(PrimitiveType::Int64) => Ok(Box::new(to_primitive::<i64>(json_col, data_type))),
        Primitive(PrimitiveType::Int128) => Ok(Box::new(to_decimal(json_col, data_type))),
        Primitive(PrimitiveType::Int256) => Ok(Box::new(to_decimal256(json_col, data_type))),
        Primitive(PrimitiveType::DaysMs) => Ok(Box::new(to_primitive_days_ms(json_col, data_type))),
        Primitive(PrimitiveType::MonthDayNano) => {
            Ok(Box::new(to_primitive_months_days_ns(json_col, data_type)))
        }
        Primitive(PrimitiveType::UInt8) => Ok(Box::new(to_primitive::<u8>(json_col, data_type))),
        Primitive(PrimitiveType::UInt16) => Ok(Box::new(to_primitive::<u16>(json_col, data_type))),
        Primitive(PrimitiveType::UInt32) => Ok(Box::new(to_primitive::<u32>(json_col, data_type))),
        Primitive(PrimitiveType::UInt64) => Ok(Box::new(to_primitive::<u64>(json_col, data_type))),
        Primitive(PrimitiveType::Float16) => todo!(),
        Primitive(PrimitiveType::Float32) => Ok(Box::new(to_primitive::<f32>(json_col, data_type))),
        Primitive(PrimitiveType::Float64) => Ok(Box::new(to_primitive::<f64>(json_col, data_type))),
        Binary => Ok(to_binary::<i32>(json_col, data_type)),
        LargeBinary => Ok(to_binary::<i64>(json_col, data_type)),
        Utf8 => Ok(to_utf8::<i32>(json_col, data_type)),
        LargeUtf8 => Ok(to_utf8::<i64>(json_col, data_type)),
        FixedSizeBinary => {
            let validity = to_validity(&json_col.validity);

            let values = json_col
                .data
                .as_ref()
                .unwrap()
                .iter()
                .flat_map(|value| value.as_str().map(|x| hex::decode(x).unwrap()).unwrap())
                .collect();
            Ok(Box::new(FixedSizeBinaryArray::new(
                data_type, values, validity,
            )))
        }
        List => to_list::<i32>(json_col, data_type, field, dictionaries),
        LargeList => to_list::<i64>(json_col, data_type, field, dictionaries),
        FixedSizeList => {
            let validity = to_validity(&json_col.validity);

            let (child_field, _) = FixedSizeListArray::get_child_and_size(&data_type);

            let children = &json_col.children.as_ref().unwrap()[0];
            let values = to_array(
                child_field.data_type().clone(),
                &field.fields[0],
                children,
                dictionaries,
            )?;

            Ok(Box::new(FixedSizeListArray::new(
                data_type, values, validity,
            )))
        }
        Struct => {
            let validity = to_validity(&json_col.validity);

            let fields = StructArray::get_fields(&data_type);

            let values = fields
                .iter()
                .zip(json_col.children.as_ref().unwrap())
                .zip(field.fields.iter())
                .map(|((field, col), ipc_field)| {
                    to_array(field.data_type().clone(), ipc_field, col, dictionaries)
                })
                .collect::<Result<Vec<_>>>()?;

            let array = StructArray::new(data_type, values, validity);
            Ok(Box::new(array))
        }
        Dictionary(key_type) => {
            match_integer_type!(key_type, |$T| {
                to_dictionary::<$T>(data_type, field, json_col, dictionaries)
            })
        }
        Union => {
            let fields = UnionArray::get_fields(&data_type);
            let fields = fields
                .iter()
                .zip(json_col.children.as_ref().unwrap())
                .zip(field.fields.iter())
                .map(|((field, col), ipc_field)| {
                    to_array(field.data_type().clone(), ipc_field, col, dictionaries)
                })
                .collect::<Result<Vec<_>>>()?;

            let types = json_col
                .type_id
                .as_ref()
                .map(|x| {
                    x.iter()
                        .map(|value| match value {
                            Value::Number(x) => {
                                x.as_i64().and_then(num_traits::cast::<i64, i8>).unwrap()
                            }
                            Value::String(x) => x.parse::<i8>().ok().unwrap(),
                            _ => {
                                panic!()
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();

            let offsets = json_col
                .offset
                .as_ref()
                .map(|x| {
                    Some(
                        x.iter()
                            .map(|value| match value {
                                Value::Number(x) => {
                                    x.as_i64().and_then(num_traits::cast::<i64, i32>).unwrap()
                                }
                                _ => panic!(),
                            })
                            .collect::<Vec<_>>()
                            .into(),
                    )
                })
                .unwrap_or_default();

            let array = UnionArray::new(data_type, types, fields, offsets);
            Ok(Box::new(array))
        }
        Map => to_map(json_col, data_type, field, dictionaries),
    }
}

/// Deserializes a [`ArrowJsonBatch`] to a [`Chunk`]
pub fn deserialize_chunk(
    schema: &Schema,
    ipc_fields: &[IpcField],
    json_batch: &ArrowJsonBatch,
    json_dictionaries: &AHashMap<i64, ArrowJsonDictionaryBatch>,
) -> Result<Chunk<Box<dyn Array>>> {
    let arrays = schema
        .fields
        .iter()
        .zip(&json_batch.columns)
        .zip(ipc_fields.iter())
        .map(|((field, json_col), ipc_field)| {
            to_array(
                field.data_type().clone(),
                ipc_field,
                json_col,
                json_dictionaries,
            )
        })
        .collect::<Result<_>>()?;

    Chunk::try_new(arrays)
}
