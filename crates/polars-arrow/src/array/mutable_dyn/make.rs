use polars_error::polars_bail;

use super::nested::*;
use crate::array::*;
use crate::datatypes::{IntegerType, *};
use crate::types::Offset;

pub fn make_mutable_array_dyn(
    dtype: &ArrowDataType,
    capacity: usize,
    dictionary_keys: Option<&Vec<String>>,
) -> PolarsResult<Box<dyn MutableArray>> {
    Ok(match dtype.to_physical_type() {
        PhysicalType::Boolean => {
            Box::new(MutableBooleanArray::with_capacity(capacity)) as Box<dyn MutableArray>
        },
        PhysicalType::Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            Box::new(MutablePrimitiveArray::<$T>::with_capacity(capacity).to(dtype.clone()))
                as Box<dyn MutableArray>
        }),
        PhysicalType::Binary => {
            Box::new(MutableBinaryArray::<i32>::with_capacity(capacity)) as Box<dyn MutableArray>
        },
        PhysicalType::LargeBinary => {
            Box::new(MutableBinaryArray::<i64>::with_capacity(capacity)) as Box<dyn MutableArray>
        },
        PhysicalType::Utf8 => {
            Box::new(MutableUtf8Array::<i32>::with_capacity(capacity)) as Box<dyn MutableArray>
        },
        PhysicalType::LargeUtf8 => {
            Box::new(MutableUtf8Array::<i64>::with_capacity(capacity)) as Box<dyn MutableArray>
        },
        PhysicalType::BinaryView => {
            Box::new(MutableBinaryViewArray::<[u8]>::with_capacity(capacity))
                as Box<dyn MutableArray>
        },
        PhysicalType::Utf8View => Box::new(MutableBinaryViewArray::<str>::with_capacity(capacity))
            as Box<dyn MutableArray>,
        _ => match dtype {
            ArrowDataType::List(inner) => {
                let values = make_mutable_array_dyn(inner.dtype(), 0, None)?;
                Box::new(DynMutableListArray::<i32>::new_from(
                    values,
                    dtype.clone(),
                    capacity,
                )) as Box<dyn MutableArray>
            },
            ArrowDataType::LargeList(inner) => {
                let values = make_mutable_array_dyn(inner.dtype(), 0, None)?;
                Box::new(DynMutableListArray::<i64>::new_from(
                    values,
                    dtype.clone(),
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
                    .map(|field| make_mutable_array_dyn(field.dtype(), capacity, None))
                    .collect::<PolarsResult<Vec<_>>>()?;
                Box::new(DynMutableStructArray::new(values, dtype.clone())) as Box<dyn MutableArray>
            },
            ArrowDataType::Dictionary(key_type, value_type, _) => {
                if let Some(keys) = dictionary_keys {
                    // NOTE: This is for compatibility with the previous Avro-internal method
                    let values = Utf8Array::<i32>::from_slice(keys);
                    Box::new(FixedItemsUtf8Dictionary::with_capacity(values, capacity))
                        as Box<dyn MutableArray>
                } else {
                    let mut arr = match value_type.to_physical_type() {
                        PhysicalType::Utf8 => make_mutable_dictionary_type::<i32>(key_type),
                        PhysicalType::LargeUtf8 => make_mutable_dictionary_type::<i64>(key_type),
                        _ => {
                            polars_bail!(nyi = "unsupported dictionary value type {value_type:#?}")
                        },
                    };
                    arr.reserve(capacity);
                    arr
                }
            },
            other => {
                polars_bail!(nyi = "cannot make dynamic mutable array for {other:#?}")
            },
        },
    })
}

fn make_mutable_dictionary_type<O: Offset>(key_type: &IntegerType) -> Box<dyn MutableArray> {
    match *key_type {
        IntegerType::Int8 => Box::new(MutableDictionaryArray::<i8, MutableUtf8Array<O>>::new())
            as Box<dyn MutableArray>,
        IntegerType::Int16 => Box::new(MutableDictionaryArray::<i16, MutableUtf8Array<O>>::new())
            as Box<dyn MutableArray>,
        IntegerType::Int32 => Box::new(MutableDictionaryArray::<i32, MutableUtf8Array<O>>::new())
            as Box<dyn MutableArray>,
        IntegerType::Int64 => Box::new(MutableDictionaryArray::<i64, MutableUtf8Array<O>>::new())
            as Box<dyn MutableArray>,
        IntegerType::Int128 => Box::new(MutableDictionaryArray::<i128, MutableUtf8Array<O>>::new())
            as Box<dyn MutableArray>,
        IntegerType::UInt8 => Box::new(MutableDictionaryArray::<u8, MutableUtf8Array<O>>::new())
            as Box<dyn MutableArray>,
        IntegerType::UInt16 => Box::new(MutableDictionaryArray::<u16, MutableUtf8Array<O>>::new())
            as Box<dyn MutableArray>,
        IntegerType::UInt32 => Box::new(MutableDictionaryArray::<u32, MutableUtf8Array<O>>::new())
            as Box<dyn MutableArray>,
        IntegerType::UInt64 => Box::new(MutableDictionaryArray::<u64, MutableUtf8Array<O>>::new())
            as Box<dyn MutableArray>,
    }
}
