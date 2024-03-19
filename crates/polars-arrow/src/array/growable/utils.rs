use polars_utils::slice::GetSaferUnchecked;

use crate::array::Array;
use crate::bitmap::MutableBitmap;
use crate::offset::Offset;

#[inline]
pub(super) unsafe fn extend_offset_values<O: Offset>(
    buffer: &mut Vec<u8>,
    offsets: &[O],
    values: &[u8],
    start: usize,
    len: usize,
) {
    let start_values = offsets.get_unchecked_release(start).to_usize();
    let end_values = offsets.get_unchecked_release(start + len).to_usize();
    let new_values = &values.get_unchecked_release(start_values..end_values);
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
                // SAFETY: invariant offset + length <= slice.len()
                unsafe {
                    mutable_validity.extend_from_slice_unchecked(slice, start + offset, len);
                }
            },
        }
    }
}

pub(super) fn extend_validity_copies(
    mutable_validity: &mut Option<MutableBitmap>,
    array: &dyn Array,
    start: usize,
    len: usize,
    copies: usize,
) {
    if let Some(mutable_validity) = mutable_validity {
        match array.validity() {
            None => mutable_validity.extend_constant(len * copies, true),
            Some(validity) => {
                debug_assert!(start + len <= validity.len());
                let (slice, offset, _) = validity.as_slice();
                // SAFETY: invariant offset + length <= slice.len()
                for _ in 0..copies {
                    unsafe {
                        mutable_validity.extend_from_slice_unchecked(slice, start + offset, len);
                    }
                }
            },
        }
    }
}
