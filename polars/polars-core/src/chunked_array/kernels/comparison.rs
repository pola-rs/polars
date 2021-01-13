use super::utils::combine_bitmaps_or;
use crate::prelude::*;
use arrow::array::{
    Array, ArrayData, BooleanArray, BooleanBufferBuilder, BufferBuilderTrait, LargeStringArray,
    PrimitiveArray,
};
use arrow::error::Result;

/// Helper function to perform boolean lambda function on values from two arrays, this
/// version does not attempt to use SIMD.
macro_rules! compare_op {
    ($left: expr, $right:expr, $op:expr) => {{
        debug_assert_eq!($left.len(), $right.len());

        let null_bit_buffer = combine_bitmaps_or($left, $right);

        let mut result = BooleanBufferBuilder::new($left.len());
        for i in 0..$left.len() {
            result.append($op($left.value(i), $right.value(i)))?;
        }

        let data = ArrayData::new(
            ArrowDataType::Boolean,
            $left.len(),
            None,
            null_bit_buffer,
            0,
            vec![result.finish()],
            vec![],
        );
        Ok(PrimitiveArray::<BooleanType>::from(Arc::new(data)))
    }};
}

macro_rules! compare_op_scalar {
    ($left: expr, $right:expr, $op:expr) => {{
        let null_bit_buffer = $left.data().null_buffer().cloned();
        let mut result = BooleanBufferBuilder::new($left.len());
        for i in 0..$left.len() {
            result.append($op($left.value(i), $right))?;
        }

        let data = ArrayData::new(
            ArrowDataType::Boolean,
            $left.len(),
            None,
            null_bit_buffer,
            0,
            vec![result.finish()],
            vec![],
        );
        Ok(PrimitiveArray::<BooleanType>::from(Arc::new(data)))
    }};
}

pub fn eq_utf8(left: &LargeStringArray, right: &LargeStringArray) -> Result<BooleanArray> {
    compare_op!(left, right, |a, b| a == b)
}

pub fn eq_utf8_scalar(left: &LargeStringArray, right: &str) -> Result<BooleanArray> {
    compare_op_scalar!(left, right, |a, b| a == b)
}

pub fn neq_utf8(left: &LargeStringArray, right: &LargeStringArray) -> Result<BooleanArray> {
    compare_op!(left, right, |a, b| a != b)
}

pub fn neq_utf8_scalar(left: &LargeStringArray, right: &str) -> Result<BooleanArray> {
    compare_op_scalar!(left, right, |a, b| a != b)
}

pub fn lt_utf8(left: &LargeStringArray, right: &LargeStringArray) -> Result<BooleanArray> {
    compare_op!(left, right, |a, b| a < b)
}

pub fn lt_utf8_scalar(left: &LargeStringArray, right: &str) -> Result<BooleanArray> {
    compare_op_scalar!(left, right, |a, b| a < b)
}

pub fn lt_eq_utf8(left: &LargeStringArray, right: &LargeStringArray) -> Result<BooleanArray> {
    compare_op!(left, right, |a, b| a <= b)
}

pub fn lt_eq_utf8_scalar(left: &LargeStringArray, right: &str) -> Result<BooleanArray> {
    compare_op_scalar!(left, right, |a, b| a <= b)
}

pub fn gt_utf8(left: &LargeStringArray, right: &LargeStringArray) -> Result<BooleanArray> {
    compare_op!(left, right, |a, b| a > b)
}

pub fn gt_utf8_scalar(left: &LargeStringArray, right: &str) -> Result<BooleanArray> {
    compare_op_scalar!(left, right, |a, b| a > b)
}

pub fn gt_eq_utf8(left: &LargeStringArray, right: &LargeStringArray) -> Result<BooleanArray> {
    compare_op!(left, right, |a, b| a >= b)
}

pub fn gt_eq_utf8_scalar(left: &LargeStringArray, right: &str) -> Result<BooleanArray> {
    compare_op_scalar!(left, right, |a, b| a >= b)
}
