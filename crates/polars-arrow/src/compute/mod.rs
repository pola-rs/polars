use arrow::array::PrimitiveArray;
use arrow::datatypes::DataType;
use arrow::types::NativeType;

use crate::utils::combine_validities_and;

pub mod arithmetics;
pub mod arity;
pub mod bitwise;
#[cfg(feature = "compute")]
pub mod cast;
#[cfg(feature = "dtype-decimal")]
pub mod decimal;
pub mod take;
pub mod tile;

#[inline]
pub fn binary_mut<T, D, F>(
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<D>,
    data_type: DataType,
    mut op: F,
) -> PrimitiveArray<T>
where
    T: NativeType,
    D: NativeType,
    F: FnMut(T, D) -> T,
{
    assert_eq!(lhs.len(), rhs.len());
    let validity = combine_validities_and(lhs.validity(), rhs.validity());

    let values = lhs
        .values()
        .iter()
        .zip(rhs.values().iter())
        .map(|(l, r)| op(*l, *r))
        .collect::<Vec<_>>()
        .into();

    PrimitiveArray::<T>::new(data_type, values, validity)
}

#[inline]
pub fn unary_mut<I, F, O>(
    array: &PrimitiveArray<I>,
    mut op: F,
    data_type: DataType,
) -> PrimitiveArray<O>
where
    I: NativeType,
    O: NativeType,
    F: FnMut(I) -> O,
{
    let values = array.values().iter().map(|v| op(*v)).collect::<Vec<_>>();

    PrimitiveArray::<O>::new(data_type, values.into(), array.validity().cloned())
}
