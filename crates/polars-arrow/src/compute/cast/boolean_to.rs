use polars_error::PolarsResult;

use super::{ArrayFromIter, BinaryViewArray, Utf8ViewArray};
use crate::array::{Array, BooleanArray, PrimitiveArray};
use crate::types::NativeType;

pub(super) fn boolean_to_primitive_dyn<T>(array: &dyn Array) -> PolarsResult<Box<dyn Array>>
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

pub fn boolean_to_utf8view(from: &BooleanArray) -> Utf8ViewArray {
    unsafe { boolean_to_binaryview(from).to_utf8view_unchecked() }
}

pub(super) fn boolean_to_utf8view_dyn(array: &dyn Array) -> PolarsResult<Box<dyn Array>> {
    let array = array.as_any().downcast_ref().unwrap();
    Ok(boolean_to_utf8view(array).boxed())
}

/// Casts the [`BooleanArray`] to a [`BinaryArray`], casting trues to `"1"` and falses to `"0"`
pub fn boolean_to_binaryview(from: &BooleanArray) -> BinaryViewArray {
    let iter = from.iter().map(|opt_b| match opt_b {
        Some(true) => Some("true".as_bytes()),
        Some(false) => Some("false".as_bytes()),
        None => None,
    });
    BinaryViewArray::arr_from_iter_trusted(iter)
}

pub(super) fn boolean_to_binaryview_dyn(array: &dyn Array) -> PolarsResult<Box<dyn Array>> {
    let array = array.as_any().downcast_ref().unwrap();
    Ok(boolean_to_binaryview(array).boxed())
}
