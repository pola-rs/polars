use polars_arrow::bitmap::Bitmap;
use polars_buffer::SharedStorage;
use polars_utils::float::IsFloat;

fn chunk_has_special<T, F: Fn(&T) -> bool>(arr: &[T; 64], is_special: F) -> bool {
    // Same autovectorization trick as chunk_has_nan in nan.rs.
    let mut has_special = false;
    for i in 0..32 {
        has_special |= is_special(&arr[i]) | is_special(&arr[i + 32]);
    }
    has_special
}

fn chunk_special_mask<T, F: Fn(&T) -> bool>(arr: &[T; 64], is_special: F) -> u64 {
    let mut mask = 0;
    for (i, v) in arr.iter().enumerate() {
        mask |= (is_special(v) as u64) << i;
    }
    mask
}

/// Returns the first i for which is_special(&slice[i]) is true, if any.
fn first_special_idx<T, F: Fn(&T) -> bool + Copy>(slice: &[T], is_special: F) -> Option<usize> {
    let mut offset = 0;
    let (chunks, last_chunk) = slice.as_chunks::<64>();
    for chunk in chunks {
        if chunk_has_special(chunk, is_special) {
            let offset_in_chunk = chunk_special_mask(chunk, is_special).trailing_zeros() as usize;
            return Some(offset + offset_in_chunk);
        }
        offset += 64;
    }
    last_chunk.iter().position(is_special).map(|i| offset + i)
}

/// Returns a bitmap where bitmap[i] = !is_special(&slice[i]), or the opposite if
/// `invert` is set. If None is returned, none of the elements are special.
fn not_special_impl<T, F: Fn(&T) -> bool + Copy>(
    slice: &[T],
    is_special: F,
    invert: bool,
) -> Option<Bitmap> {
    let invert_mask = if invert { u64::MAX } else { 0 };
    let first_idx = first_special_idx(slice, is_special)?;
    let no_special_chunks = first_idx / 64;
    let mut words = Vec::with_capacity(slice.len().div_ceil(64));
    let mut unset_bits = 0;
    words.resize(no_special_chunks, u64::MAX ^ invert_mask);

    let (chunks, last_chunk) = slice.as_chunks::<64>();
    let mut chunk_idx = no_special_chunks;
    while chunk_idx < chunks.len() {
        let special_mask = chunk_special_mask(&chunks[chunk_idx], is_special);
        words.push(!special_mask ^ invert_mask);
        unset_bits += special_mask.count_ones() as usize;
        chunk_idx += 1;

        if special_mask == 0 {
            // Special values are probably rare, fast-path for skipping.
            while chunk_idx < chunks.len() && !chunk_has_special(&chunks[chunk_idx], is_special) {
                words.push(u64::MAX ^ invert_mask);
                chunk_idx += 1
            }
        }
    }

    let mut last_word = 0;
    for (i, v) in last_chunk.iter().enumerate() {
        let special = is_special(v);
        last_word |= (!special as u64) << i;
        unset_bits += special as usize;
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

/// Returns a bitmap, where bitmap[i] = slice[i].is_finite(). If None is returned
/// all of the elements are finite.
pub fn is_finite<T: IsFloat>(slice: &[T]) -> Option<Bitmap> {
    assert!(T::is_float());
    not_special_impl(slice, |v: &T| !v.is_finite(), false)
}

/// Returns a bitmap, where bitmap[i] = slice[i] is +inf or -inf. If None is
/// returned none of the elements are infinite.
pub fn is_infinite<T: IsFloat>(slice: &[T]) -> Option<Bitmap> {
    assert!(T::is_float());
    not_special_impl(slice, |v: &T| !v.is_finite() & !v.is_nan(), true)
}
