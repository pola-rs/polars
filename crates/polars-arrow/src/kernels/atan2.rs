use arrow::array::PrimitiveArray;
use arrow::compute::arity::binary;
use arrow::types::NativeType;
use num_traits::Float;

pub fn atan2<T: NativeType>(
    arr_1: &PrimitiveArray<T>,
    arr_2: &PrimitiveArray<T>,
) -> PrimitiveArray<T>
where
    T: Float,
{
    binary(arr_1, arr_2, arr_1.data_type().clone(), |a, b| a.atan2(b))
}
