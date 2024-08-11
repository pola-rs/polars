use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, PrimitiveArray, Utf8Array, Utf8ViewArray,
};
use arrow::scalar::{BinaryScalar, BinaryViewScalar, BooleanScalar, PrimitiveScalar, Scalar};

use crate::min_max::MinMaxKernel;

macro_rules! call_op {
    ($T:ty, $scalar:ty, $arr:expr, $op:path) => {{
        let arr: &$T = $arr.as_any().downcast_ref().unwrap();
        $op(arr).map(|v| Box::new(<$scalar>::new(Some(v))) as Box<dyn Scalar>)
    }};
    (dt: $T:ty, $scalar:ty, $arr:expr, $op:path) => {{
        let arr: &$T = $arr.as_any().downcast_ref().unwrap();
        $op(arr)
            .map(|v| Box::new(<$scalar>::new(arr.data_type().clone(), Some(v))) as Box<dyn Scalar>)
    }};
    ($T:ty, $scalar:ty, $arr:expr, $op:path, ret_two) => {{
        let arr: &$T = $arr.as_any().downcast_ref().unwrap();
        $op(arr).map(|(l, r)| {
            (
                Box::new(<$scalar>::new(Some(l))) as Box<dyn Scalar>,
                Box::new(<$scalar>::new(Some(r))) as Box<dyn Scalar>,
            )
        })
    }};
    (dt: $T:ty, $scalar:ty, $arr:expr, $op:path, ret_two) => {{
        let arr: &$T = $arr.as_any().downcast_ref().unwrap();
        $op(arr).map(|(l, r)| {
            (
                Box::new(<$scalar>::new(arr.data_type().clone(), Some(l))) as Box<dyn Scalar>,
                Box::new(<$scalar>::new(arr.data_type().clone(), Some(r))) as Box<dyn Scalar>,
            )
        })
    }};
}

macro_rules! call {
    ($arr:expr, $op:path$(, $variant:ident)?) => {{
        let arr = $arr;

        use arrow::datatypes::{PhysicalType as PH, PrimitiveType as PR};
        use PrimitiveArray as PArr;
        use PrimitiveScalar as PScalar;
        match arr.data_type().to_physical_type() {
            PH::Boolean => call_op!(BooleanArray, BooleanScalar, arr, $op$(, $variant)?),
            PH::Primitive(PR::Int8) => call_op!(dt: PArr<i8>, PScalar<i8>, arr, $op$(, $variant)?),
            PH::Primitive(PR::Int16) => call_op!(dt: PArr<i16>, PScalar<i16>, arr, $op$(, $variant)?),
            PH::Primitive(PR::Int32) => call_op!(dt: PArr<i32>, PScalar<i32>, arr, $op$(, $variant)?),
            PH::Primitive(PR::Int64) => call_op!(dt: PArr<i64>, PScalar<i64>, arr, $op$(, $variant)?),
            PH::Primitive(PR::Int128) => call_op!(dt: PArr<i128>, PScalar<i128>, arr, $op$(, $variant)?),
            PH::Primitive(PR::UInt8) => call_op!(dt: PArr<u8>, PScalar<u8>, arr, $op$(, $variant)?),
            PH::Primitive(PR::UInt16) => call_op!(dt: PArr<u16>, PScalar<u16>, arr, $op$(, $variant)?),
            PH::Primitive(PR::UInt32) => call_op!(dt: PArr<u32>, PScalar<u32>, arr, $op$(, $variant)?),
            PH::Primitive(PR::UInt64) => call_op!(dt: PArr<u64>, PScalar<u64>, arr, $op$(, $variant)?),
            PH::Primitive(PR::UInt128) => call_op!(dt: PArr<u128>, PScalar<u128>, arr, $op$(, $variant)?),
            PH::Primitive(PR::Float32) => call_op!(dt: PArr<f32>, PScalar<f32>, arr, $op$(, $variant)?),
            PH::Primitive(PR::Float64) => call_op!(dt: PArr<f64>, PScalar<f64>, arr, $op$(, $variant)?),

            PH::BinaryView => call_op!(BinaryViewArray, BinaryViewScalar<[u8]>, arr, $op$(, $variant)?),
            PH::Utf8View => call_op!(Utf8ViewArray, BinaryViewScalar<str>, arr, $op$(, $variant)?),

            PH::Binary => call_op!(BinaryArray<i32>, BinaryScalar<i32>, arr, $op$(, $variant)?),
            PH::LargeBinary => call_op!(BinaryArray<i64>, BinaryScalar<i64>, arr, $op$(, $variant)?),
            PH::Utf8 => call_op!(Utf8Array<i32>, BinaryScalar<i32>, arr, $op$(, $variant)?),
            PH::LargeUtf8 => call_op!(Utf8Array<i64>, BinaryScalar<i64>, arr, $op$(, $variant)?),

            _ => todo!("Dynamic MinMax is not yet implemented for {:?}", arr.data_type()),
        }
    }};
}

pub fn dyn_array_min_ignore_nan(arr: &dyn Array) -> Option<Box<dyn Scalar>> {
    call!(arr, MinMaxKernel::min_ignore_nan_kernel)
}

pub fn dyn_array_max_ignore_nan(arr: &dyn Array) -> Option<Box<dyn Scalar>> {
    call!(arr, MinMaxKernel::max_ignore_nan_kernel)
}

pub fn dyn_array_min_propagate_nan(arr: &dyn Array) -> Option<Box<dyn Scalar>> {
    call!(arr, MinMaxKernel::min_propagate_nan_kernel)
}

pub fn dyn_array_max_propagate_nan(arr: &dyn Array) -> Option<Box<dyn Scalar>> {
    call!(arr, MinMaxKernel::max_propagate_nan_kernel)
}

pub fn dyn_array_min_max_propagate_nan(
    arr: &dyn Array,
) -> Option<(Box<dyn Scalar>, Box<dyn Scalar>)> {
    call!(arr, MinMaxKernel::min_max_propagate_nan_kernel, ret_two)
}
