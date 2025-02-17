use crate::array::Array;
use crate::bitmap::BitmapBuilder;
use crate::offset::Offset;

#[inline]
pub(super) unsafe fn extend_offset_values<O: Offset>(
    buffer: &mut Vec<u8>,
    offsets: &[O],
    values: &[u8],
    start: usize,
    len: usize,
) {
    let start_values = offsets.get_unchecked(start).to_usize();
    let end_values = offsets.get_unchecked(start + len).to_usize();
    let new_values = &values.get_unchecked(start_values..end_values);
    buffer.extend_from_slice(new_values);
}

pub(super) fn prepare_validity(use_validity: bool, capacity: usize) -> Option<BitmapBuilder> {
    if use_validity {
        Some(BitmapBuilder::with_capacity(capacity))
    } else {
        None
    }
}

pub(super) fn extend_validity(
    mutable_validity: &mut Option<BitmapBuilder>,
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
                mutable_validity.extend_from_slice(slice, start + offset, len);
            },
        }
    }
}

pub(super) fn extend_validity_copies(
    mutable_validity: &mut Option<BitmapBuilder>,
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
                for _ in 0..copies {
                    mutable_validity.extend_from_slice(slice, start + offset, len);
                }
            },
        }
    }
}
