use crate::array::PrimitiveArray;
use crate::types::NativeType;

pub(super) fn equal<T: NativeType>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> bool {
    lhs.dtype() == rhs.dtype() && lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
