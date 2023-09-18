use crate::{array::Array, bitmap::MutableBitmap, offset::Offset};

// function used to extend nulls from arrays. This function's lifetime is bound to the array
// because it reads nulls from it.
pub(super) type ExtendNullBits<'a> = Box<dyn Fn(&mut MutableBitmap, usize, usize) + 'a>;

pub(super) fn build_extend_null_bits(array: &dyn Array, use_validity: bool) -> ExtendNullBits {
    if let Some(bitmap) = array.validity() {
        Box::new(move |validity, start, len| {
            debug_assert!(start + len <= bitmap.len());
            let (slice, offset, _) = bitmap.as_slice();
            // safety: invariant offset + length <= slice.len()
            unsafe {
                validity.extend_from_slice_unchecked(slice, start + offset, len);
            }
        })
    } else if use_validity {
        Box::new(|validity, _, len| {
            validity.extend_constant(len, true);
        })
    } else {
        Box::new(|_, _, _| {})
    }
}

#[inline]
pub(super) fn extend_offset_values<O: Offset>(
    buffer: &mut Vec<u8>,
    offsets: &[O],
    values: &[u8],
    start: usize,
    len: usize,
) {
    let start_values = offsets[start].to_usize();
    let end_values = offsets[start + len].to_usize();
    let new_values = &values[start_values..end_values];
    buffer.extend_from_slice(new_values);
}
