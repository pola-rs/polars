use crate::arrow_field::{ENUMERANTS_METADATA_KEY, ENUMERANTS_METADATA_SEPARATOR};
use polars_arrow::array::{
    MutableArray, MutableBinaryArray, MutableBooleanArray, MutableDictionaryArray,
    MutableListArray, MutablePrimitiveArray, MutableStructArray, MutableUtf8Array,
};
use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField};

pub fn make_mutable_arrays(fields: &[ArrowField], length: usize) -> Vec<Box<dyn MutableArray>> {
    fields
        .iter()
        .map(|field| map_mutable_array(field, length))
        .collect()
}

fn map_mutable_array(field: &ArrowField, length: usize) -> Box<dyn MutableArray> {
    match field.data_type() {
        ArrowDataType::Boolean => Box::new(MutableBooleanArray::with_capacity(length)),
        ArrowDataType::Int8 => Box::new(MutablePrimitiveArray::<i8>::with_capacity(length)),
        ArrowDataType::Int16 => Box::new(MutablePrimitiveArray::<i16>::with_capacity(length)),
        ArrowDataType::Int32 => Box::new(MutablePrimitiveArray::<i32>::with_capacity(length)),
        ArrowDataType::Int64 => Box::new(MutablePrimitiveArray::<i64>::with_capacity(length)),
        ArrowDataType::UInt8 => Box::new(MutablePrimitiveArray::<u8>::with_capacity(length)),
        ArrowDataType::UInt16 => Box::new(MutablePrimitiveArray::<u16>::with_capacity(length)),
        ArrowDataType::UInt32 => Box::new(MutablePrimitiveArray::<u32>::with_capacity(length)),
        ArrowDataType::UInt64 => Box::new(MutablePrimitiveArray::<u64>::with_capacity(length)),
        ArrowDataType::Float32 => Box::new(MutablePrimitiveArray::<f32>::with_capacity(length)),
        ArrowDataType::Float64 => Box::new(MutablePrimitiveArray::<f64>::with_capacity(length)),
        ArrowDataType::Utf8 => Box::new(MutableUtf8Array::<i32>::with_capacity(length)),
        ArrowDataType::Binary => Box::new(MutableBinaryArray::<i32>::with_capacity(length)),
        ArrowDataType::Dictionary(_, _, _) => {
            let mut enumerants = MutableUtf8Array::<i32>::new();
            for enumerant in field
                .metadata
                .get(ENUMERANTS_METADATA_KEY)
                .expect("Dictionary fields should have enumerants in metadata")
                .split(ENUMERANTS_METADATA_SEPARATOR)
            {
                enumerants.push(Some(enumerant));
            }
            let mut array =
                MutableDictionaryArray::<u16, MutableUtf8Array<i32>>::from_values(enumerants)
                    .unwrap();
            array.reserve(length);
            Box::new(array)
        }
        ArrowDataType::Struct(inner_fields) => {
            let mut inner_arrays: Vec<Box<dyn MutableArray>> = Vec::new();
            for inner_field in inner_fields.iter() {
                inner_arrays.push(map_mutable_array(inner_field, length));
            }
            Box::new(MutableStructArray::new(
                field.data_type().clone(),
                inner_arrays,
            ))
        }
        ArrowDataType::List(inner_field) => {
            let inner_array = map_mutable_array(inner_field, length);
            Box::new(MutableListArray::<i32, _>::new_from(
                inner_array,
                field.data_type().clone(),
                length,
            ))
        }
        _ => panic!("unsupported type"),
    }
}
