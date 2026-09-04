#![allow(clippy::eq_op)] // We use x != x to detect NaN generically.

use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_array::{ArrayRepr, PlBitmap, PlBooleanArray, PlPrimitiveArray};
use polars_buffer::SharedStorage;
use polars_utils::float::IsFloat;

fn chunk_has_nan<T: PartialEq>(arr: &[T; 64]) -> bool {
    // This has some hackery to improve autovectorization.
    let mut has_nan = false;
    for i in 0..32 {
        has_nan |= (arr[i] != arr[i]) | (arr[i + 32] != arr[i + 32]);
    }
    has_nan
}

fn chunk_nan_mask<T: PartialEq>(arr: &[T; 64]) -> u64 {
    let mut mask = 0;
    for (i, v) in arr.iter().enumerate() {
        mask |= ((v != v) as u64) << i;
    }
    mask
}

/// Returns the first i for which slice[i].is_nan() is true, if any.
pub fn first_nan_idx<T: PartialEq + IsFloat>(slice: &[T]) -> Option<usize> {
    assert!(T::is_float());
    let mut offset = 0;
    let (chunks, last_chunk) = slice.as_chunks::<64>();
    for chunk in chunks {
        if chunk_has_nan(chunk) {
            let offset_in_chunk = chunk_nan_mask(chunk).trailing_zeros() as usize;
            return Some(offset + offset_in_chunk);
        }
        offset += 64;
    }
    last_chunk.iter().position(|x| x != x).map(|i| offset + i)
}

/// Returns a bitmap, where bitmap[i] = slice[i].is_nan(). If None is returned
/// none of the elements are NaN.
pub fn is_nan_slice<T: PartialEq + IsFloat>(slice: &[T]) -> Option<Bitmap> {
    nan_mask_slice(slice, true)
}

/// Returns a bitmap, where bitmap[i] = !slice[i].is_nan(). If None is returned
/// none of the elements are NaN.
pub fn is_not_nan_slice<T: PartialEq + IsFloat>(slice: &[T]) -> Option<Bitmap> {
    nan_mask_slice(slice, false)
}

/// Returns a bitmap where bit `i` says whether `slice[i]` is NaN, if `nan_is_set`, and whether it
/// is not, otherwise. `None` stands for a slice that holds no NaN at all, whose mask is therefore
/// `nan_is_set` nowhere and `!nan_is_set` everywhere.
fn nan_mask_slice<T: PartialEq + IsFloat>(slice: &[T], nan_is_set: bool) -> Option<Bitmap> {
    assert!(T::is_float());
    let invert = nan_is_set;
    let invert_mask = if invert { u64::MAX } else { 0 };
    let first_idx = first_nan_idx(slice)?;
    let no_nan_chunks = first_idx / 64;
    let mut words = Vec::with_capacity(slice.len().div_ceil(64));
    let mut unset_bits = 0;
    words.resize(no_nan_chunks, u64::MAX ^ invert_mask);

    let (chunks, last_chunk) = slice.as_chunks::<64>();
    let mut chunk_idx = no_nan_chunks;
    while chunk_idx < chunks.len() {
        let nan_mask = chunk_nan_mask(&chunks[chunk_idx]);
        words.push(!nan_mask ^ invert_mask);
        unset_bits += nan_mask.count_ones() as usize;
        chunk_idx += 1;

        if nan_mask == 0 {
            // NaNs are probably rare, fast-path for skipping.
            while chunk_idx < chunks.len() && !chunk_has_nan(&chunks[chunk_idx]) {
                words.push(u64::MAX ^ invert_mask);
                chunk_idx += 1
            }
        }
    }

    let mut last_word = 0;
    for (i, v) in last_chunk.iter().enumerate() {
        let is_nan = v != v;
        last_word |= (!is_nan as u64) << i;
        unset_bits += is_nan as usize;
    }
    words.push(last_word ^ invert_mask);

    if invert {
        unset_bits = slice.len() - unset_bits;
    }

    let storage = SharedStorage::from_vec(words)
        .try_transmute::<u8>()
        .ok()
        .unwrap();
    let bitmap = unsafe { Bitmap::from_inner_unchecked(storage, 0, slice.len(), Some(unset_bits)) };
    Some(bitmap)
}

/// Returns a mask that is set where `array` is NaN.
///
/// A null element is NaN nowhere and not-NaN nowhere either: the mask carries `array`'s validity,
/// so the answer at a null element is null in turn.
pub fn is_nan<T: NativeType + IsFloat>(array: &PlPrimitiveArray<T>) -> PlBooleanArray {
    nan_mask(array, true)
}

/// Returns a mask that is set where `array` is not NaN; see [`is_nan`].
pub fn is_not_nan<T: NativeType + IsFloat>(array: &PlPrimitiveArray<T>) -> PlBooleanArray {
    nan_mask(array, false)
}

fn nan_mask<T: NativeType + IsFloat>(
    array: &PlPrimitiveArray<T>,
    nan_is_set: bool,
) -> PlBooleanArray {
    // A scalar values buffer holds the one value every element reads: it is tested once, and the
    // one answer stands for the whole chunk, in `O(1)` memory.
    let values = match array.values_repr() {
        ArrayRepr::Scalar(value) => {
            PlBitmap::new_scalar((value != value) == nan_is_set, array.len())
        },
        ArrayRepr::Flat(slice) => {
            match nan_mask_slice(slice, nan_is_set) {
                Some(mask) => PlBitmap::new(mask, array.len()),
                // No element is NaN, so the whole mask is the one answer that holds for all of
                // them, which needs no slot per element either.
                None => PlBitmap::new_scalar(!nan_is_set, array.len()),
            }
        },
    };

    PlBooleanArray::from_pl_bitmap(values).with_validity(array.validity().map(PlBitmap::from))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one value a scalar chunk holds is tested once, and the mask repeats the answer rather
    /// than holding a bit per element.
    #[test]
    fn a_repeated_value_is_tested_once() {
        let nans = PlPrimitiveArray::new_scalar(f64::NAN, 100);
        assert!(is_nan(&nans).values_are_scalar());
        assert_eq!(is_nan(&nans), PlBooleanArray::new_scalar(true, 100));
        assert_eq!(is_not_nan(&nans), PlBooleanArray::new_scalar(false, 100));

        let numbers = PlPrimitiveArray::new_scalar(1.0f64, 100);
        assert!(is_nan(&numbers).values_are_scalar());
        assert_eq!(is_nan(&numbers), PlBooleanArray::new_scalar(false, 100));
        assert_eq!(is_not_nan(&numbers), PlBooleanArray::new_scalar(true, 100));
    }

    /// A chunk laid out one value per element is tested one element at a time, and the mask comes
    /// out with the validity it went in with.
    #[test]
    fn every_element_is_tested() {
        let arr = PlPrimitiveArray::from_iter([Some(1.0f64), None, Some(f64::NAN)]);
        assert_eq!(
            is_nan(&arr),
            PlBooleanArray::from_iter([Some(false), None, Some(true)]),
        );
        assert_eq!(
            is_not_nan(&arr),
            PlBooleanArray::from_iter([Some(true), None, Some(false)]),
        );
    }

    /// A flat chunk that holds no NaN at all needs no slot per element to say so.
    #[test]
    fn a_chunk_without_nan_answers_once() {
        let arr = PlPrimitiveArray::from_vec(vec![1.0f64, 2.0, 3.0]);
        assert!(is_nan(&arr).values_are_scalar());
        assert_eq!(is_nan(&arr), PlBooleanArray::new_scalar(false, 3));
        assert_eq!(is_not_nan(&arr), PlBooleanArray::new_scalar(true, 3));
    }

    #[test]
    fn an_empty_chunk_is_tested_nowhere() {
        let empty = PlPrimitiveArray::<f64>::new_empty();
        assert!(is_nan(&empty).is_empty());
        assert!(is_not_nan(&empty).is_empty());
    }
}
