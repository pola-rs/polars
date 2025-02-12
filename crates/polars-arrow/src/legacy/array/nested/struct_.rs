use std::any::Any;

use polars_error::{polars_bail, PolarsResult};

use super::dyn_array_push;
use crate::array::{
    DynMutableListArray, DynMutableStructArray, MutableArray, MutableBinaryArray,
    MutableBinaryViewArray, MutableBooleanArray, MutablePrimitiveArray, MutableUtf8Array,
};
use crate::datatypes::PhysicalType;
use crate::scalar::{
    BinaryScalar, BinaryViewScalar, BooleanScalar, ListScalar, PrimitiveScalar, Scalar,
    StructScalar, Utf8Scalar,
};
use crate::types::{days_ms, f16, i256, months_days_ns, PrimitiveType};

macro_rules! dyn_struct_scalar_push {
    ($arr:expr, $idx:expr, $arr_ty:ty, $scalar:expr, $scalar_ty:ty) => {{
        let dst = $arr
            .mut_values($idx)
            .as_mut_any()
            .downcast_mut::<$arr_ty>()
            .unwrap();
        let src = $scalar.as_any().downcast_ref::<$scalar_ty>().unwrap();
        dst.push(src.value());
    }};
}

macro_rules! dyn_struct_scalar_push_primitive {
    ($arr:expr, $idx:expr, $scalar:expr, $ty:ty) => {{
        let dst = $arr
            .mut_values($idx)
            .as_mut_any()
            .downcast_mut::<MutablePrimitiveArray<$ty>>()
            .unwrap();
        let src = $scalar
            .as_any()
            .downcast_ref::<PrimitiveScalar<$ty>>()
            .unwrap();
        dst.push(*src.value());
    }};
}

macro_rules! dyn_struct_scalar_push_list {
    ($arr:expr, $idx:expr, $scalar:expr, $offset:ty, $dtype:expr) => {{
        let dst = $arr
            .mut_values($idx)
            .as_mut_any()
            .downcast_mut::<DynMutableListArray<$offset>>()
            .unwrap();
        let src = $scalar
            .as_any()
            .downcast_ref::<ListScalar<$offset>>()
            .unwrap();
        let value = if src.is_valid() {
            Some(src.values())
        } else {
            None
        };
        dyn_array_push(dst, value, $dtype)?;
    }};
}

pub fn dyn_struct_array_push<T: Any>(arr: &mut dyn MutableArray, value: &T) -> PolarsResult<()> {
    let mut arr = arr
        .as_mut_any()
        .downcast_mut::<DynMutableStructArray>()
        .unwrap();
    let values = (value as &dyn Any)
        .downcast_ref::<Vec<Box<dyn Scalar>>>()
        .unwrap();

    for (idx, scalar) in values.iter().enumerate() {
        dyn_struct_array_push_field(&mut arr, idx, scalar)?;
    }

    arr.try_push_valid()
}

pub fn dyn_struct_array_push_field(
    arr: &mut DynMutableStructArray,
    idx: usize,
    scalar: &Box<dyn Scalar>,
) -> PolarsResult<()> {
    let dtype = scalar.dtype();
    match dtype.to_physical_type() {
        PhysicalType::Primitive(primitive) => match primitive {
            PrimitiveType::Int8 => dyn_struct_scalar_push_primitive!(arr, idx, scalar, i8),
            PrimitiveType::Int16 => dyn_struct_scalar_push_primitive!(arr, idx, scalar, i16),
            PrimitiveType::Int32 => dyn_struct_scalar_push_primitive!(arr, idx, scalar, i32),
            PrimitiveType::Int64 => dyn_struct_scalar_push_primitive!(arr, idx, scalar, i64),
            PrimitiveType::Int128 => dyn_struct_scalar_push_primitive!(arr, idx, scalar, i128),
            PrimitiveType::Int256 => dyn_struct_scalar_push_primitive!(arr, idx, scalar, i256),
            PrimitiveType::UInt8 => dyn_struct_scalar_push_primitive!(arr, idx, scalar, u8),
            PrimitiveType::UInt16 => dyn_struct_scalar_push_primitive!(arr, idx, scalar, u16),
            PrimitiveType::UInt32 => dyn_struct_scalar_push_primitive!(arr, idx, scalar, u32),
            PrimitiveType::UInt64 => dyn_struct_scalar_push_primitive!(arr, idx, scalar, u64),
            PrimitiveType::UInt128 => dyn_struct_scalar_push_primitive!(arr, idx, scalar, u128),
            PrimitiveType::Float16 => dyn_struct_scalar_push_primitive!(arr, idx, scalar, f16),
            PrimitiveType::Float32 => dyn_struct_scalar_push_primitive!(arr, idx, scalar, f32),
            PrimitiveType::Float64 => dyn_struct_scalar_push_primitive!(arr, idx, scalar, f64),
            PrimitiveType::DaysMs => {
                dyn_struct_scalar_push_primitive!(arr, idx, scalar, days_ms)
            },
            PrimitiveType::MonthDayNano => {
                dyn_struct_scalar_push_primitive!(arr, idx, scalar, months_days_ns)
            },
        },
        PhysicalType::Boolean => {
            dyn_struct_scalar_push!(arr, idx, MutableBooleanArray, scalar, BooleanScalar)
        },
        PhysicalType::Binary => {
            dyn_struct_scalar_push!(arr, idx, MutableBinaryArray<i32>, scalar, BinaryScalar<i32>)
        },
        PhysicalType::LargeBinary => {
            dyn_struct_scalar_push!(arr, idx, MutableBinaryArray<i64>, scalar, BinaryScalar<i64>)
        },
        PhysicalType::BinaryView => {
            dyn_struct_scalar_push!(
                arr,
                idx,
                MutableBinaryViewArray<[u8]>,
                scalar,
                BinaryViewScalar<[u8]>
            )
        },
        PhysicalType::Utf8 => {
            dyn_struct_scalar_push!(arr, idx, MutableUtf8Array<i32>, scalar, Utf8Scalar<i32>)
        },
        PhysicalType::LargeUtf8 => {
            dyn_struct_scalar_push!(arr, idx, MutableUtf8Array<i64>, scalar, Utf8Scalar<i64>)
        },
        PhysicalType::Utf8View => {
            dyn_struct_scalar_push!(
                arr,
                idx,
                MutableBinaryViewArray<str>,
                scalar,
                BinaryViewScalar<str>
            )
        },
        PhysicalType::List => {
            dyn_struct_scalar_push_list!(arr, idx, scalar, i32, dtype)
        },
        PhysicalType::LargeList => {
            dyn_struct_scalar_push_list!(arr, idx, scalar, i64, dtype)
        },
        PhysicalType::Struct => {
            let dst = arr
                .mut_values(idx)
                .as_mut_any()
                .downcast_mut::<DynMutableStructArray>()
                .unwrap();
            let src = scalar.as_any().downcast_ref::<StructScalar>().unwrap();
            let value = if src.is_valid() {
                // NOTE: We have to clone here as we require ownership of the values
                Some(src.values().iter().map(|x| x.clone()).collect::<Vec<_>>())
            } else {
                None
            };
            dyn_array_push(dst, value.as_ref(), dtype)?;
        },
        _ => polars_bail!(nyi = "cannot push dynamic struct value of type {dtype:?}"),
    }
    Ok(())
}
