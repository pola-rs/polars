use std::any::Any;

use polars_error::{polars_bail, PolarsResult};

use super::dyn_array_push;
use crate::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, DynMutableListArray, DynMutableStructArray,
    ListArray, MutableArray, MutableBinaryArray, MutableBinaryViewArray, MutableBooleanArray,
    MutablePrimitiveArray, MutableUtf8Array, PrimitiveArray, StructArray, Utf8Array, Utf8ViewArray,
};
use crate::datatypes::PhysicalType;
use crate::types::{days_ms, f16, i256, months_days_ns, Offset, PrimitiveType};

macro_rules! dyn_list_array_push_item {
    ($arr:expr, $arr_ty:ty, $value:expr, $value_ty:ty, $dtype:expr) => {{
        let dst = $arr
            .mut_values()
            .as_mut_any()
            .downcast_mut::<$arr_ty>()
            .unwrap();
        let src = $value.as_any().downcast_ref::<$value_ty>().unwrap();
        for item in src.iter() {
            dyn_array_push(dst, item.as_ref(), $dtype)?;
        }
    }};
}

macro_rules! dyn_list_array_extend {
    ($arr:expr, $arr_ty:ty, $value:expr, $value_ty:ty) => {{
        let dst = $arr
            .mut_values()
            .as_mut_any()
            .downcast_mut::<$arr_ty>()
            .unwrap();
        let src = $value.as_any().downcast_ref::<$value_ty>().unwrap();
        dst.extend_trusted_len(src.iter());
    }};
}

macro_rules! dyn_list_array_extend_primitive {
    ($arr:expr, $value:expr, $ty:ty) => {{
        dyn_list_array_extend!(
            $arr,
            MutablePrimitiveArray<$ty>,
            $value,
            PrimitiveArray<$ty>
        );
    }};
}

pub fn dyn_list_array_push<O: Offset, T: Any>(
    arr: &mut dyn MutableArray,
    value: &T,
) -> PolarsResult<()> {
    let mut arr = arr
        .as_mut_any()
        .downcast_mut::<DynMutableListArray<O>>()
        .unwrap();
    let value = (value as &dyn Any)
        .downcast_ref::<Box<dyn Array>>()
        .unwrap();

    let dtype = value.dtype();
    match dtype.to_physical_type() {
        PhysicalType::Primitive(primitive) => match primitive {
            PrimitiveType::Int8 => dyn_list_array_extend_primitive!(&mut arr, value, i8),
            PrimitiveType::Int16 => dyn_list_array_extend_primitive!(&mut arr, value, i16),
            PrimitiveType::Int32 => dyn_list_array_extend_primitive!(&mut arr, value, i32),
            PrimitiveType::Int64 => dyn_list_array_extend_primitive!(&mut arr, value, i64),
            PrimitiveType::Int128 => dyn_list_array_extend_primitive!(&mut arr, value, i128),
            PrimitiveType::Int256 => dyn_list_array_extend_primitive!(&mut arr, value, i256),
            PrimitiveType::UInt8 => dyn_list_array_extend_primitive!(&mut arr, value, u8),
            PrimitiveType::UInt16 => dyn_list_array_extend_primitive!(&mut arr, value, u16),
            PrimitiveType::UInt32 => dyn_list_array_extend_primitive!(&mut arr, value, u32),
            PrimitiveType::UInt64 => dyn_list_array_extend_primitive!(&mut arr, value, u64),
            PrimitiveType::UInt128 => dyn_list_array_extend_primitive!(&mut arr, value, u128),
            PrimitiveType::Float16 => dyn_list_array_extend_primitive!(&mut arr, value, f16),
            PrimitiveType::Float32 => dyn_list_array_extend_primitive!(&mut arr, value, f32),
            PrimitiveType::Float64 => dyn_list_array_extend_primitive!(&mut arr, value, f64),
            PrimitiveType::DaysMs => {
                dyn_list_array_extend_primitive!(&mut arr, value, days_ms)
            },
            PrimitiveType::MonthDayNano => {
                dyn_list_array_extend_primitive!(&mut arr, value, months_days_ns)
            },
        },
        PhysicalType::Boolean => {
            dyn_list_array_extend!(arr, MutableBooleanArray, value, BooleanArray)
        },
        PhysicalType::Binary => {
            dyn_list_array_extend!(arr, MutableBinaryArray<i32>, value, BinaryArray<i32>)
        },
        PhysicalType::LargeBinary => {
            dyn_list_array_extend!(arr, MutableBinaryArray<i64>, value, BinaryArray<i64>)
        },
        PhysicalType::BinaryView => {
            dyn_list_array_extend!(arr, MutableBinaryViewArray<[u8]>, value, BinaryViewArray)
        },
        PhysicalType::Utf8 => {
            dyn_list_array_extend!(arr, MutableUtf8Array<i32>, value, Utf8Array<i32>)
        },
        PhysicalType::LargeUtf8 => {
            dyn_list_array_extend!(arr, MutableUtf8Array<i64>, value, Utf8Array<i64>)
        },
        PhysicalType::Utf8View => {
            dyn_list_array_extend!(arr, MutableBinaryViewArray<str>, value, Utf8ViewArray)
        },
        PhysicalType::List => {
            dyn_list_array_push_item!(arr, DynMutableListArray<i32>, value, ListArray<i32>, dtype)
        },
        PhysicalType::LargeList => {
            dyn_list_array_push_item!(arr, DynMutableListArray<i64>, value, ListArray<i64>, dtype)
        },
        PhysicalType::Struct => {
            dyn_list_array_push_item!(arr, DynMutableStructArray, value, StructArray, dtype)
        },
        _ => polars_bail!(nyi = "cannot push dynamic list value of type {dtype:?}"),
    };

    arr.try_push_valid()
}
