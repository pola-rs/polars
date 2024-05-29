use num_traits::Float;

use crate::array::PrimitiveArray;
use crate::compute::arity::binary;
use crate::types::NativeType;

pub fn atan2<T>(arr_1: &PrimitiveArray<T>, arr_2: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: Float + NativeType,
{
    binary(arr_1, arr_2, arr_1.data_type().clone(), |a, b| a.atan2(b))
}
