use crate::array::PrimitiveArray;
use crate::datatypes::ArrowDataType;
use crate::compute::utils::combine_validities_and;
use crate::types::NativeType;

pub mod arithmetics;
pub mod bitwise;
#[cfg(feature = "compute_cast")]
pub mod cast;
#[cfg(feature = "dtype-decimal")]
pub mod decimal;
pub mod take;
pub mod tile;

#[inline]
pub fn binary_mut<T, D, F>(
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<D>,
    data_type: ArrowDataType,
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
    data_type: ArrowDataType,
) -> PrimitiveArray<O>
where
    I: NativeType,
    O: NativeType,
    F: FnMut(I) -> O,
{
    let values = array.values().iter().map(|v| op(*v)).collect::<Vec<_>>();

    PrimitiveArray::<O>::new(data_type, values.into(), array.validity().cloned())
}
