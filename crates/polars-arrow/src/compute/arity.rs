//! Defines kernels suitable to perform operations to primitive arrays.

use polars_error::PolarsResult;

use super::utils::{check_same_len, combine_validities_and};
use crate::array::PrimitiveArray;
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::datatypes::ArrowDataType;
use crate::types::NativeType;

/// Applies an unary and infallible function to a [`PrimitiveArray`].
///
/// This is the /// fastest way to perform an operation on a [`PrimitiveArray`] when the benefits
/// of a vectorized operation outweighs the cost of branching nulls and non-nulls.
///
/// # Implementation
/// This will apply the function for all values, including those on null slots.
/// This implies that the operation must be infallible for any value of the
/// corresponding type or this function may panic.
#[inline]
pub fn unary<I, F, O>(
    array: &PrimitiveArray<I>,
    op: F,
    data_type: ArrowDataType,
) -> PrimitiveArray<O>
where
    I: NativeType,
    O: NativeType,
    F: Fn(I) -> O,
{
    let values = array.values().iter().map(|v| op(*v)).collect::<Vec<_>>();

    PrimitiveArray::<O>::new(data_type, values.into(), array.validity().cloned())
}

/// Version of unary that checks for errors in the closure used to create the
/// buffer
pub fn try_unary<I, F, O>(
    array: &PrimitiveArray<I>,
    op: F,
    data_type: ArrowDataType,
) -> PolarsResult<PrimitiveArray<O>>
where
    I: NativeType,
    O: NativeType,
    F: Fn(I) -> PolarsResult<O>,
{
    let values = array
        .values()
        .iter()
        .map(|v| op(*v))
        .collect::<PolarsResult<Vec<_>>>()?
        .into();

    Ok(PrimitiveArray::<O>::new(
        data_type,
        values,
        array.validity().cloned(),
    ))
}

/// Version of unary that returns an array and bitmap. Used when working with
/// overflowing operations
pub fn unary_with_bitmap<I, F, O>(
    array: &PrimitiveArray<I>,
    op: F,
    data_type: ArrowDataType,
) -> (PrimitiveArray<O>, Bitmap)
where
    I: NativeType,
    O: NativeType,
    F: Fn(I) -> (O, bool),
{
    let mut mut_bitmap = MutableBitmap::with_capacity(array.len());

    let values = array
        .values()
        .iter()
        .map(|v| {
            let (res, over) = op(*v);
            mut_bitmap.push(over);
            res
        })
        .collect::<Vec<_>>()
        .into();

    (
        PrimitiveArray::<O>::new(data_type, values, array.validity().cloned()),
        mut_bitmap.into(),
    )
}

/// Version of unary that creates a mutable bitmap that is used to keep track
/// of checked operations. The resulting bitmap is compared with the array
/// bitmap to create the final validity array.
pub fn unary_checked<I, F, O>(
    array: &PrimitiveArray<I>,
    op: F,
    data_type: ArrowDataType,
) -> PrimitiveArray<O>
where
    I: NativeType,
    O: NativeType,
    F: Fn(I) -> Option<O>,
{
    let mut mut_bitmap = MutableBitmap::with_capacity(array.len());

    let values = array
        .values()
        .iter()
        .map(|v| match op(*v) {
            Some(val) => {
                mut_bitmap.push(true);
                val
            },
            None => {
                mut_bitmap.push(false);
                O::default()
            },
        })
        .collect::<Vec<_>>()
        .into();

    // The validity has to be checked against the bitmap created during the
    // creation of the values with the iterator. If an error was found during
    // the iteration, then the validity is changed to None to mark the value
    // as Null
    let bitmap: Bitmap = mut_bitmap.into();
    let validity = combine_validities_and(array.validity(), Some(&bitmap));

    PrimitiveArray::<O>::new(data_type, values, validity)
}

/// Applies a binary operations to two primitive arrays.
///
/// This is the fastest way to perform an operation on two primitive array when the benefits of a
/// vectorized operation outweighs the cost of branching nulls and non-nulls.
///
/// # Errors
/// This function errors iff the arrays have a different length.
///
/// # Implementation
/// This will apply the function for all values, including those on null slots.
/// This implies that the operation must be infallible for any value of the
/// corresponding type.
/// The types of the arrays are not checked with this operation. The closure
/// "op" needs to handle the different types in the arrays. The datatype for the
/// resulting array has to be selected by the implementer of the function as
/// an argument for the function.
#[inline]
pub fn binary<T, D, F>(
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<D>,
    data_type: ArrowDataType,
    op: F,
) -> PrimitiveArray<T>
where
    T: NativeType,
    D: NativeType,
    F: Fn(T, D) -> T,
{
    check_same_len(lhs, rhs).unwrap();

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

/// Version of binary that checks for errors in the closure used to create the
/// buffer
pub fn try_binary<T, D, F>(
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<D>,
    data_type: ArrowDataType,
    op: F,
) -> PolarsResult<PrimitiveArray<T>>
where
    T: NativeType,
    D: NativeType,
    F: Fn(T, D) -> PolarsResult<T>,
{
    check_same_len(lhs, rhs)?;

    let validity = combine_validities_and(lhs.validity(), rhs.validity());

    let values = lhs
        .values()
        .iter()
        .zip(rhs.values().iter())
        .map(|(l, r)| op(*l, *r))
        .collect::<PolarsResult<Vec<_>>>()?
        .into();

    Ok(PrimitiveArray::<T>::new(data_type, values, validity))
}

/// Version of binary that returns an array and bitmap. Used when working with
/// overflowing operations
pub fn binary_with_bitmap<T, D, F>(
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<D>,
    data_type: ArrowDataType,
    op: F,
) -> (PrimitiveArray<T>, Bitmap)
where
    T: NativeType,
    D: NativeType,
    F: Fn(T, D) -> (T, bool),
{
    check_same_len(lhs, rhs).unwrap();

    let validity = combine_validities_and(lhs.validity(), rhs.validity());

    let mut mut_bitmap = MutableBitmap::with_capacity(lhs.len());

    let values = lhs
        .values()
        .iter()
        .zip(rhs.values().iter())
        .map(|(l, r)| {
            let (res, over) = op(*l, *r);
            mut_bitmap.push(over);
            res
        })
        .collect::<Vec<_>>()
        .into();

    (
        PrimitiveArray::<T>::new(data_type, values, validity),
        mut_bitmap.into(),
    )
}

/// Version of binary that creates a mutable bitmap that is used to keep track
/// of checked operations. The resulting bitmap is compared with the array
/// bitmap to create the final validity array.
pub fn binary_checked<T, D, F>(
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<D>,
    data_type: ArrowDataType,
    op: F,
) -> PrimitiveArray<T>
where
    T: NativeType,
    D: NativeType,
    F: Fn(T, D) -> Option<T>,
{
    check_same_len(lhs, rhs).unwrap();

    let mut mut_bitmap = MutableBitmap::with_capacity(lhs.len());

    let values = lhs
        .values()
        .iter()
        .zip(rhs.values().iter())
        .map(|(l, r)| match op(*l, *r) {
            Some(val) => {
                mut_bitmap.push(true);
                val
            },
            None => {
                mut_bitmap.push(false);
                T::default()
            },
        })
        .collect::<Vec<_>>()
        .into();

    let bitmap: Bitmap = mut_bitmap.into();
    let validity = combine_validities_and(lhs.validity(), rhs.validity());

    // The validity has to be checked against the bitmap created during the
    // creation of the values with the iterator. If an error was found during
    // the iteration, then the validity is changed to None to mark the value
    // as Null
    let validity = combine_validities_and(validity.as_ref(), Some(&bitmap));

    PrimitiveArray::<T>::new(data_type, values, validity)
}
