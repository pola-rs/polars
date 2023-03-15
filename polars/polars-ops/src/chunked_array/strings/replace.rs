use arrow::array::Utf8Array;
use arrow::offset::OffsetsBuffer;

// ensure the offsets are corrected in case of sliced arrays
fn correct_offsets(offsets: OffsetsBuffer<i64>, start: i64) -> OffsetsBuffer<i64> {
    if start != 0 {
        let offsets_buf: Vec<i64> = offsets.iter().map(|o| *o - start).collect();
        return unsafe { OffsetsBuffer::new_unchecked(offsets_buf.into()) };
    }
    offsets
}

pub(super) fn replace_lit_single_char(arr: &Utf8Array<i64>, pat: u8, val: u8) -> Utf8Array<i64> {
    let values = arr.values();
    let offsets = arr.offsets().clone();
    let validity = arr.validity().cloned();
    let start = offsets[0] as usize;
    let end = (offsets[offsets.len() - 1]) as usize;

    let mut values = values.as_slice()[start..end].to_vec();
    for byte in values.iter_mut() {
        if *byte == pat {
            *byte = val;
        }
    }
    // ensure the offsets are corrected in case of sliced arrays
    let offsets = correct_offsets(offsets, start as i64);
    unsafe { Utf8Array::new_unchecked(arr.data_type().clone(), offsets, values.into(), validity) }
}

pub(super) fn replace_lit_n_char(
    arr: &Utf8Array<i64>,
    n: usize,
    pat: u8,
    val: u8,
) -> Utf8Array<i64> {
    let values = arr.values();
    let offsets = arr.offsets().clone();
    let validity = arr.validity().cloned();
    let start = offsets[0] as usize;
    let end = (offsets[offsets.len() - 1]) as usize;

    let mut values = values.as_slice()[start..end].to_vec();
    let mut offsets_iter = offsets.iter();
    // ignore the first
    let _ = offsets_iter.next().unwrap();

    let mut end = *offsets_iter.next().unwrap() as usize - 1;
    let mut count = 0;
    for (i, byte) in values.iter_mut().enumerate() {
        if *byte == pat && count < n {
            *byte = val;
            count += 1;
        };
        if i == end {
            // reset the count as we entered a new string region
            count = 0;

            // set the end of this string region
            // safety: invariant of Utf8Array tells us that there is a next offset.
            if let Some(next) = offsets_iter.next() {
                end = *next as usize - 1;
            }
        }
    }
    // ensure the offsets are corrected in case of sliced arrays
    let offsets = correct_offsets(offsets, start as i64);
    unsafe { Utf8Array::new_unchecked(arr.data_type().clone(), offsets, values.into(), validity) }
}
