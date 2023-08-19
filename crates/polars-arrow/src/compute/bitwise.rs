use std::ops::{BitAnd, BitOr, BitXor};

use arrow::array::PrimitiveArray;
use arrow::compute::arity::binary;
use arrow::types::NativeType;

pub fn bitand<T: NativeType>(a: &PrimitiveArray<T>, b: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: BitAnd<T, Output = T>,
{
    binary(a, b, a.data_type().clone(), |a, b| a.bitand(b))
}

pub fn bitor<T: NativeType>(a: &PrimitiveArray<T>, b: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: BitOr<T, Output = T>,
{
    binary(a, b, a.data_type().clone(), |a, b| a.bitor(b))
}

pub fn bitxor<T: NativeType>(a: &PrimitiveArray<T>, b: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: BitXor<T, Output = T>,
{
    binary(a, b, a.data_type().clone(), |a, b| a.bitxor(b))
}
