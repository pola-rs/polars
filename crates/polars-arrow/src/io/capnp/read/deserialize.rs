use crate::io::capnp::read::array::make_mutable_arrays;
use crate::io::capnp::read::reader::{get_schema, read_from_capnp_struct};
use crate::io::capnp::read::zipped_field::{zip_fields, ZippedField};
use capnp::Error;
use capnp::{dynamic_struct, dynamic_value};
use crate::array::{
    Array, MutableArray, MutableBinaryArray, MutableBooleanArray, MutableDictionaryArray,
    MutableListArray, MutablePrimitiveArray, MutableStructArray, MutableUtf8Array, TryPush,
};
use crate::record_batch::RecordBatch;
use crate::datatypes::{ArrowDataType, Field as ArrowField};
use polars_error::PolarsResult;

macro_rules! push_value {
    ($a:expr, $t:ty, $v:expr) => {{
        let array = $a.as_mut_any().downcast_mut::<$t>().unwrap();
        array.push(Some($v));
    }};
}

pub fn deserialize(
    messages: &[dynamic_value::Reader],
    arrow_fields: &[ArrowField],
) -> PolarsResult<RecordBatch<Box<dyn Array>>> {
    let mut arrays = make_mutable_arrays(arrow_fields, messages.len());
    let zipped_fields = zip_fields(get_schema(messages), arrow_fields).unwrap();
    for message in messages {
        let iter = arrays.iter_mut().zip(zipped_fields.iter());
        for (array, field) in iter {
            deserialize_struct_field(field, message, array.as_mut(), true);
        }
    }
    Ok(RecordBatch::try_new(
        arrays
            .iter_mut()
            .map(|array| array.as_box())
            .collect(),
    ))
}

// Get the capnp value from the struct.
// If is_valid is false, then this struct or a parent field is a member of a union
// and is not valid for this message. We ignore the capnp value and will push a null
// value to the arrow array and any inner arrays of nested types (lists and structs).
fn deserialize_struct_field(
    zipped_field: &ZippedField,
    capnp_struct: &dynamic_value::Reader,
    array: &mut dyn MutableArray,
    is_valid: bool,
) {
    if is_valid {
        match read_from_capnp_struct(
            &capnp_struct.downcast::<dynamic_struct::Reader>(),
            zipped_field.capnp_field(),
        ) {
            Some(capnp_value) => deserialize_value(zipped_field, &capnp_value, array, true),
            None => deserialize_value(zipped_field, capnp_struct, array, false),
        }
    } else {
        deserialize_value(zipped_field, capnp_struct, array, false);
    }
}

fn deserialize_value(
    zipped_field: &ZippedField,
    capnp_value: &dynamic_value::Reader,
    array: &mut dyn MutableArray,
    is_valid: bool,
) {
    match (zipped_field.arrow_field().data_type(), is_valid) {
        (ArrowDataType::Boolean, true) => {
            push_value!(array, MutableBooleanArray, capnp_value.downcast())
        }
        (ArrowDataType::Int8, true) => {
            push_value!(array, MutablePrimitiveArray<i8>, capnp_value.downcast())
        }
        (ArrowDataType::Int16, true) => {
            push_value!(array, MutablePrimitiveArray<i16>, capnp_value.downcast())
        }
        (ArrowDataType::Int32, true) => {
            push_value!(array, MutablePrimitiveArray<i32>, capnp_value.downcast())
        }
        (ArrowDataType::Int64, true) => {
            push_value!(array, MutablePrimitiveArray<i64>, capnp_value.downcast())
        }
        (ArrowDataType::UInt8, true) => {
            push_value!(array, MutablePrimitiveArray<u8>, capnp_value.downcast())
        }
        (ArrowDataType::UInt16, true) => {
            push_value!(array, MutablePrimitiveArray<u16>, capnp_value.downcast())
        }
        (ArrowDataType::UInt32, true) => {
            push_value!(array, MutablePrimitiveArray<u32>, capnp_value.downcast())
        }
        (ArrowDataType::UInt64, true) => {
            push_value!(array, MutablePrimitiveArray<u64>, capnp_value.downcast())
        }
        (ArrowDataType::Float32, true) => {
            push_value!(array, MutablePrimitiveArray<f32>, capnp_value.downcast())
        }
        (ArrowDataType::Float64, true) => {
            push_value!(array, MutablePrimitiveArray<f64>, capnp_value.downcast())
        }
        (ArrowDataType::Binary, true) => push_value!(
            array,
            MutableBinaryArray<i32>,
            capnp_value.downcast::<capnp::data::Reader>()
        ),
        (ArrowDataType::Utf8, true) => push_value!(
            array,
            MutableUtf8Array<i32>,
            capnp_value
                .downcast::<capnp::text::Reader>()
                .to_string()
                .unwrap()
        ),
        (ArrowDataType::Dictionary(_, _, _), true) => {
            let e = capnp_value.downcast::<capnp::dynamic_value::Enum>();
            match e.get_enumerant().unwrap() {
                Some(enumerant) => {
                    let value = enumerant.get_proto().get_name().unwrap().to_str().unwrap();
                    let array = array
                        .as_mut_any()
                        .downcast_mut::<MutableDictionaryArray<u16, MutableUtf8Array<i32>>>()
                        .unwrap();
                    array.try_push(Some(value)).unwrap()
                }
                None => array.push_null(),
            }
        }
        (ArrowDataType::Struct(_), _) => {
            let array = array
                .as_mut_any()
                .downcast_mut::<MutableStructArray>()
                .unwrap();
            for (inner_array, inner_field) in array
                .mut_values()
                .iter_mut()
                .zip(zipped_field.inner_fields().iter())
            {
                deserialize_struct_field(inner_field, capnp_value, inner_array.as_mut(), is_valid);
            }
            array.push(is_valid);
        }
        (ArrowDataType::List(_), _) => {
            type M = Box<dyn MutableArray>;
            let array = array
                .as_mut_any()
                .downcast_mut::<MutableListArray<i32, M>>()
                .unwrap();
            let inner_array: &mut dyn MutableArray = array.mut_values();
            if is_valid {
                let list = capnp_value.downcast::<capnp::dynamic_list::Reader>();
                for inner_value in list.iter() {
                    deserialize_value(
                        zipped_field.inner_field(),
                        &inner_value.unwrap(),
                        inner_array,
                        is_valid,
                    );
                }
            } else {
                deserialize_value(
                    zipped_field.inner_field(),
                    capnp_value,
                    inner_array,
                    is_valid,
                );
            }
            array.try_push_valid().unwrap();
        }
        _ => array.push_null(),
    }
}
