use crate::array::Array;
use crate::bitmap::MutableBitmap;
use crate::offset::Offset;

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

pub(super) fn prepare_validity(use_validity: bool, capacity: usize) -> Option<MutableBitmap> {
    if use_validity {
        Some(MutableBitmap::with_capacity(capacity))
    } else {
        None
    }
}

pub(super) fn extend_validity(
    mutable_validity: &mut Option<MutableBitmap>,
    array: &dyn Array,
    start: usize,
    len: usize,
) {
    if let Some(mutable_validity) = mutable_validity {
        match array.validity() {
            None => mutable_validity.extend_constant(len, true),
            Some(validity) => {
                debug_assert!(start + len <= validity.len());
                let (slice, offset, _) = validity.as_slice();
                // safety: invariant offset + length <= slice.len()
                unsafe {
                    mutable_validity.extend_from_slice_unchecked(slice, start + offset, len);
                }
            },
        }
    }
}
