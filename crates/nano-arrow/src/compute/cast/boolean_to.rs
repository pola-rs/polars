use crate::{
    array::{Array, BinaryArray, BooleanArray, PrimitiveArray, Utf8Array},
    error::Result,
    offset::Offset,
    types::NativeType,
};

pub(super) fn boolean_to_primitive_dyn<T>(array: &dyn Array) -> Result<Box<dyn Array>>
where
    T: NativeType + num_traits::One,
{
    let array = array.as_any().downcast_ref().unwrap();
    Ok(Box::new(boolean_to_primitive::<T>(array)))
}

/// Casts the [`BooleanArray`] to a [`PrimitiveArray`].
pub fn boolean_to_primitive<T>(from: &BooleanArray) -> PrimitiveArray<T>
where
    T: NativeType + num_traits::One,
{
    let values = from
        .values()
        .iter()
        .map(|x| if x { T::one() } else { T::default() })
        .collect::<Vec<_>>();

    PrimitiveArray::<T>::new(T::PRIMITIVE.into(), values.into(), from.validity().cloned())
}

/// Casts the [`BooleanArray`] to a [`Utf8Array`], casting trues to `"1"` and falses to `"0"`
pub fn boolean_to_utf8<O: Offset>(from: &BooleanArray) -> Utf8Array<O> {
    let iter = from.values().iter().map(|x| if x { "1" } else { "0" });
    Utf8Array::from_trusted_len_values_iter(iter)
}

pub(super) fn boolean_to_utf8_dyn<O: Offset>(array: &dyn Array) -> Result<Box<dyn Array>> {
    let array = array.as_any().downcast_ref().unwrap();
    Ok(Box::new(boolean_to_utf8::<O>(array)))
}

/// Casts the [`BooleanArray`] to a [`BinaryArray`], casting trues to `"1"` and falses to `"0"`
pub fn boolean_to_binary<O: Offset>(from: &BooleanArray) -> BinaryArray<O> {
    let iter = from.values().iter().map(|x| if x { b"1" } else { b"0" });
    BinaryArray::from_trusted_len_values_iter(iter)
}

pub(super) fn boolean_to_binary_dyn<O: Offset>(array: &dyn Array) -> Result<Box<dyn Array>> {
    let array = array.as_any().downcast_ref().unwrap();
    Ok(Box::new(boolean_to_binary::<O>(array)))
}
