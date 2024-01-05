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
    // ensure the offsets are corrected in case of sliced arrays
    let offsets = correct_offsets(offsets, start as i64);

    let mut offsets_iter = offsets.iter();
    // ignore the first
    let _ = *offsets_iter.next().unwrap();
    let mut end = None;
    // must loop to skip all null/empty values, as they all have the same offsets.
    for next in offsets_iter.by_ref() {
        // we correct offsets before, it's guaranteed to start at 0.
        if *next != 0 {
            end = Some(*next as usize - 1);
            break;
        }
    }

    let Some(mut end) = end else {
        return arr.clone();
    };

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

            // must loop to skip null/empty values, as they have the same offsets
            for next in offsets_iter.by_ref() {
                let new_end = *next as usize - 1;
                if new_end != end {
                    end = new_end;
                    break;
                }
            }
        }
    }
    unsafe { Utf8Array::new_unchecked(arr.data_type().clone(), offsets, values.into(), validity) }
}

pub(super) fn replace_lit_n_str(
    arr: &Utf8Array<i64>,
    n: usize,
    pat: &str,
    val: &str,
) -> Utf8Array<i64> {
    assert_eq!(pat.len(), val.len());
    let values = arr.values();
    let offsets = arr.offsets().clone();
    let validity = arr.validity().cloned();
    let start = offsets[0] as usize;
    let end = (offsets[offsets.len() - 1]) as usize;

    let mut values = values.as_slice()[start..end].to_vec();
    // // ensure the offsets are corrected in case of sliced arrays
    let offsets = correct_offsets(offsets, start as i64);
    let mut offsets_iter = offsets.iter();

    // overwrite previous every iter
    let mut previous = *offsets_iter.next().unwrap();

    let values_str = unsafe { std::str::from_utf8_unchecked_mut(&mut values) };
    for &end in offsets_iter {
        let substr = unsafe { values_str.get_unchecked_mut(previous as usize..end as usize) };

        for (start, part) in substr.match_indices(pat).take(n) {
            let len = part.len();
            // safety:
            // this violates the aliasing rules
            // if this become a problem we must implement our own `match_indices`
            // that works on pointers instead of references.
            unsafe {
                let bytes = std::slice::from_raw_parts_mut(
                    substr.as_bytes().as_ptr().add(start) as *mut u8,
                    len,
                );
                bytes.copy_from_slice(val.as_bytes());
            }
        }
        previous = end;
    }
    unsafe { Utf8Array::new_unchecked(arr.data_type().clone(), offsets, values.into(), validity) }
}
