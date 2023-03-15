use arrow::array::Utf8Array;
use arrow::offset::OffsetsBuffer;

pub(super) fn replace_lit_single_char(arr: &Utf8Array<i64>, pat: u8, val: u8) -> Utf8Array<i64> {
    let values = arr.values();
    let mut offsets = arr.offsets().clone();
    let validity = arr.validity().cloned();
    let start = offsets[0] as usize;
    let end = (offsets[offsets.len() - 1] + 1) as usize;

    let mut values = values.as_slice()[start..end].to_vec();
    for byte in values.iter_mut() {
        if *byte == pat {
            *byte = val;
        }
    }
    if start != 0 {
        let start = start as i64;
        let offsets_buf: Vec<i64> = offsets.iter().map(|o| *o - start).collect();
        offsets = unsafe { OffsetsBuffer::new_unchecked(offsets_buf.into()) };
    }
    unsafe { Utf8Array::new_unchecked(arr.data_type().clone(), offsets, values.into(), validity) }
}
