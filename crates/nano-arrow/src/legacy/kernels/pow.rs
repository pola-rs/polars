use num_traits::pow::Pow;

use crate::array::PrimitiveArray;
use crate::compute::arity::binary;
use crate::types::NativeType;

pub fn pow<T, F>(arr_1: &PrimitiveArray<T>, arr_2: &PrimitiveArray<F>) -> PrimitiveArray<T>
where
    T: Pow<F, Output = T> + NativeType,
    F: NativeType,
{
    binary(arr_1, arr_2, arr_1.data_type().clone(), |a, b| {
        Pow::pow(a, b)
    })
}
