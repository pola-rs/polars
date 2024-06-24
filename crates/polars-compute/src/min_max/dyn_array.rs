use arrow::array::{Array, BooleanArray, PrimitiveArray};
use arrow::scalar::{BooleanScalar, PrimitiveScalar, Scalar};

use crate::min_max::MinMaxKernel;
macro_rules! call_op {
    ($T:ty, $scalar:ty, $(=> ($($arg:expr),+),)? $arr:expr, $op:path) => {{
        let arr: &$T = $arr.as_any().downcast_ref().unwrap();
        $op(arr).map(|v| Box::new(<$scalar>::new($($($arg,)+)? Some(v))) as Box<dyn Scalar>)
    }};
}

macro_rules! call {
    ($arr:expr, $op:path) => {{
        let arr = $arr;

        use arrow::datatypes::{PhysicalType as PH, PrimitiveType as PR};
        match arr.data_type().to_physical_type() {
            PH::Boolean => call_op!(BooleanArray, BooleanScalar, arr, $op),
            PH::Primitive(PR::Int8) => call_op!(PrimitiveArray<i8>, PrimitiveScalar<i8>, => (arr.data_type().clone()), arr, $op),
            PH::Primitive(PR::Int16) => {
                call_op!(PrimitiveArray<i16>, PrimitiveScalar<i16>, => (arr.data_type().clone()), arr, $op)
            },
            PH::Primitive(PR::Int32) => {
                call_op!(PrimitiveArray<i32>, PrimitiveScalar<i32>, => (arr.data_type().clone()), arr, $op)
            },
            PH::Primitive(PR::Int64) => {
                call_op!(PrimitiveArray<i64>, PrimitiveScalar<i64>, => (arr.data_type().clone()), arr, $op)
            },
            PH::Primitive(PR::Int128) => {
                call_op!(PrimitiveArray<i128>, PrimitiveScalar<i128>, => (arr.data_type().clone()), arr, $op)
            },
            PH::Primitive(PR::UInt8) => call_op!(PrimitiveArray<u8>, PrimitiveScalar<u8>, => (arr.data_type().clone()), arr, $op),
            PH::Primitive(PR::UInt16) => {
                call_op!(PrimitiveArray<u16>, PrimitiveScalar<u16>, => (arr.data_type().clone()), arr, $op)
            },
            PH::Primitive(PR::UInt32) => {
                call_op!(PrimitiveArray<u32>, PrimitiveScalar<u32>, => (arr.data_type().clone()), arr, $op)
            },
            PH::Primitive(PR::UInt64) => {
                call_op!(PrimitiveArray<u64>, PrimitiveScalar<u64>, => (arr.data_type().clone()), arr, $op)
            },
            PH::Primitive(PR::UInt128) => {
                call_op!(PrimitiveArray<u128>, PrimitiveScalar<u128>, => (arr.data_type().clone()), arr, $op)
            },
            PH::Primitive(PR::Float32) => {
                call_op!(PrimitiveArray<f32>, PrimitiveScalar<f32>, => (arr.data_type().clone()), arr, $op)
            },
            PH::Primitive(PR::Float64) => {
                call_op!(PrimitiveArray<f64>, PrimitiveScalar<f64>, => (arr.data_type().clone()), arr, $op)
            },

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
