use crate::{array::PrimitiveArray, types::NativeType};

pub(super) fn equal<T: NativeType>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> bool {
    lhs.data_type() == rhs.data_type() && lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
