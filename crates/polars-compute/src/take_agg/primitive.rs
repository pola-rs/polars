#![allow(unsafe_op_in_unsafe_fn)]
//! Gather-and-reduce over a [`PlPrimitiveArray`].

use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use either::Either;
use num_traits::ToPrimitive;
use polars_array::PlPrimitiveArray;
use polars_utils::IdxSize;

/// The value every index gathers, where the values buffer holds a single slot.
///
/// A buffer of one slot is what an array of one element holds too, and reading it as the value it
/// repeats is right either way.
#[inline]
pub(super) fn repeated_value<T: NativeType>(arr: &PlPrimitiveArray<T>) -> Option<T> {
    arr.scalar_values()
}

/// The values buffer of `arr`, which holds one slot per element because it is not scalar.
#[inline]
pub(super) fn flat_values<T: NativeType>(arr: &PlPrimitiveArray<T>) -> &[T] {
    arr.flat_values()
        .expect("a values buffer that is not scalar holds one slot per element")
        .as_slice()
}

/// The mask of `arr` as one bit per element, or [`None`] where every element is null.
///
/// A mask that holds a single bit and has a null under it has that bit unset, so it masks out
/// every element and there is nothing left for a gather to find.
#[inline]
pub(super) fn flat_validity<T: NativeType>(arr: &PlPrimitiveArray<T>) -> Option<&Bitmap> {
    arr.validity()
        .expect("a chunk with nulls in it holds a validity mask")
        .flat_bitmap()
}

/// The values `indices` gather out of a chunk with no nulls in it.
///
/// # Safety
/// Every index must be in bounds of `arr`.
#[inline]
pub unsafe fn take_agg_no_null_primitive_iter_unchecked<
    T: NativeType + ToPrimitive,
    I: IntoIterator<Item = usize>,
>(
    arr: &PlPrimitiveArray<T>,
    indices: I,
) -> impl Iterator<Item = T> {
    debug_assert!(arr.null_count() == 0);

    match repeated_value(arr) {
        // Every index gathers the one value the buffer holds, so it is read once here rather than
        // through the buffer once per index.
        Some(value) => Either::Left(indices.into_iter().map(move |_| value)),
        None => {
            let values = flat_values(arr);
            Either::Right(
                indices
                    .into_iter()
                    .map(|idx| unsafe { *values.get_unchecked(idx) }),
            )
        },
    }
}

/// The non-null values `indices` gather out of a chunk.
///
/// # Safety
/// Every index must be in bounds of `arr`.
#[inline]
pub unsafe fn take_agg_primitive_iter_unchecked<T: NativeType, I: IntoIterator<Item = usize>>(
    arr: &PlPrimitiveArray<T>,
    indices: I,
) -> impl Iterator<Item = T> {
    let Some(validity) = flat_validity(arr) else {
        // Every element is null, so no index gathers anything.
        return Either::Left(std::iter::empty());
    };

    match repeated_value(arr) {
        Some(value) => Either::Right(Either::Left(
            indices
                .into_iter()
                .filter(move |&idx| unsafe { validity.get_bit_unchecked(idx) })
                .map(move |_| value),
        )),
        None => {
            let values = flat_values(arr);
            Either::Right(Either::Right(
                indices
                    .into_iter()
                    .filter(|&idx| unsafe { validity.get_bit_unchecked(idx) })
                    .map(|idx| unsafe { *values.get_unchecked(idx) }),
            ))
        },
    }
}

/// Folds the non-null values `indices` gather with `f`, alongside the number of nulls skipped.
///
/// Returns [`None`] where every one of the `len` indices gathered a null.
///
/// # Safety
/// Every index must be in bounds of `arr`.
#[inline]
pub unsafe fn take_agg_primitive_iter_unchecked_count_nulls<
    T: NativeType + ToPrimitive,
    I: IntoIterator<Item = usize>,
    TOut,
    F: Fn(TOut, T) -> TOut,
>(
    arr: &PlPrimitiveArray<T>,
    indices: I,
    init: TOut,
    f: F,
    len: IdxSize,
) -> Option<(TOut, IdxSize)> {
    let Some(validity) = flat_validity(arr) else {
        // Every element is null, so every one of the `len` indices gathered one.
        return None;
    };

    let repeated = repeated_value(arr);
    let values = repeated.is_none().then(|| flat_values(arr));

    let mut null_count = 0 as IdxSize;
    let out = indices.into_iter().fold(init, |acc, idx| {
        if unsafe { validity.get_bit_unchecked(idx) } {
            // One of the two is there, per `repeated_value`.
            let value = match (repeated, values) {
                (Some(value), _) => value,
                (None, Some(values)) => unsafe { *values.get_unchecked(idx) },
                (None, None) => unreachable!("the values of an array are flat or scalar"),
            };
            f(acc, value)
        } else {
            null_count += 1;
            acc
        }
    });

    if null_count == len {
        None
    } else {
        Some((out, null_count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LENGTH: usize = 6;
    const INDICES: [usize; 4] = [0, 3, 1, 5];

    /// A gather off a scalar chunk has to read the same values as one off the same chunk written
    /// out, which is what the callers used to hand these kernels.
    #[test]
    fn a_repeated_value_gathers_the_same_either_way() {
        let scalar = PlPrimitiveArray::new_scalar(7i32, LENGTH);
        let flat = PlPrimitiveArray::from_vec(vec![7i32; LENGTH]);

        let gathered = |arr: &PlPrimitiveArray<i32>| unsafe {
            take_agg_no_null_primitive_iter_unchecked(arr, INDICES).collect::<Vec<_>>()
        };
        assert_eq!(gathered(&scalar), vec![7i32; INDICES.len()]);
        assert_eq!(gathered(&scalar), gathered(&flat));
    }

    /// A mask holding one bit is either set — and then the chunk has no nulls at all — or unset,
    /// and then no index gathers anything.
    #[test]
    fn a_repeated_null_gathers_nothing() {
        let arr = PlPrimitiveArray::<i32>::new_full_null(LENGTH);
        assert_eq!(arr.null_count(), LENGTH);

        let gathered =
            unsafe { take_agg_primitive_iter_unchecked(&arr, INDICES).collect::<Vec<_>>() };
        assert!(gathered.is_empty());

        let folded = unsafe {
            take_agg_primitive_iter_unchecked_count_nulls(
                &arr,
                INDICES,
                0i32,
                |a, b| a + b,
                INDICES.len() as IdxSize,
            )
        };
        assert_eq!(folded, None);
    }

    /// The mask is read per index whatever the values look like.
    #[test]
    fn a_repeated_value_under_a_flat_mask() {
        let mask = [true, false, true, true, false, true];
        let scalar = PlPrimitiveArray::new_scalar(7i32, LENGTH)
            .with_validity(Some(mask.into_iter().collect()));
        let flat = PlPrimitiveArray::from_vec(vec![7i32; LENGTH])
            .with_validity(Some(mask.into_iter().collect()));

        let gathered = |arr: &PlPrimitiveArray<i32>| unsafe {
            take_agg_primitive_iter_unchecked(arr, INDICES).collect::<Vec<_>>()
        };
        // Index 1 is masked out, the other three are not.
        assert_eq!(gathered(&scalar), vec![7i32; 3]);
        assert_eq!(gathered(&scalar), gathered(&flat));

        let folded = |arr: &PlPrimitiveArray<i32>| unsafe {
            take_agg_primitive_iter_unchecked_count_nulls(
                arr,
                INDICES,
                0i32,
                |a, b| a + b,
                INDICES.len() as IdxSize,
            )
        };
        assert_eq!(folded(&scalar), Some((21, 1)));
        assert_eq!(folded(&scalar), folded(&flat));
    }
}
