#![allow(clippy::eq_op)] // We use x != x to detect NaN generically.

use arrow::bitmap::Bitmap;
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
pub fn is_nan<T: PartialEq + IsFloat>(slice: &[T]) -> Option<Bitmap> {
    is_not_nan_impl(slice, true)
}

/// Returns a bitmap, where bitmap[i] = !slice[i].is_nan(). If None is returned
/// none of the elements are NaN.
pub fn is_not_nan<T: PartialEq + IsFloat>(slice: &[T]) -> Option<Bitmap> {
    is_not_nan_impl(slice, false)
}

fn is_not_nan_impl<T: PartialEq + IsFloat>(slice: &[T], invert: bool) -> Option<Bitmap> {
    assert!(T::is_float());
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
