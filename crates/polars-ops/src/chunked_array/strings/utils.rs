use std::iter::repeat;
use std::ops::Deref;

use arrow::array::View;
use arrow::buffer::Buffer;
use arrow::types::NativeType;
use polars_core::prelude::StringChunked;

pub(super) fn subview(s: &str, v: View, start: usize, end: usize) -> View {
    let start = start as u32;
    let end = end as u32;
    let len = end - start;
    let offset = v.offset + start;
    if len <= 12 {
        let mut payload = [0; 16];
        payload[0..4].copy_from_slice(&len.to_le_bytes());
        if v.length <= 12 {
            payload[4..4 + len as usize]
                .copy_from_slice(&v.to_le_bytes()[4 + start as usize..4 + end as usize]);
        } else {
            payload[4..4 + len as usize]
                .copy_from_slice(&s.as_bytes()[start as usize..start as usize + len as usize]);
        }
        View::from_le_bytes(payload)
    } else {
        let mut v = if start != 0 {
            let mut payload = v.to_le_bytes();
            payload[4..8].copy_from_slice(&s.as_bytes()[start as usize..start as usize + 4]);
            View::from_le_bytes(payload)
        } else {
            v
        };
        v.length = len;
        v.offset = offset;
        v
    }
}

pub(super) fn subview_from_str(s: &str, subs: &str, v: View) -> View {
    let s_start = s.as_ptr() as usize;
    let s_end = s_start + s.len();
    let subs_start = subs.as_ptr() as usize;
    let subs_end = subs_start + subs.len();
    debug_assert!(subs_start >= s_start);
    debug_assert!(subs_end <= s_end);
    let start = subs_start - s_start;
    let end = subs_end - s_start;
    subview(s, v, start, end)
}

pub(super) fn iter_with_view_and_buffers(
    ca: &StringChunked,
) -> impl Iterator<Item = (Option<&str>, (&View, &[Buffer<u8>]))> {
    ca.downcast_iter().flat_map(|arr| {
        arr.iter()
            .zip(arr.views().iter().zip(repeat(arr.data_buffers().deref())))
    })
}
