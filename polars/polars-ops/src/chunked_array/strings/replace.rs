use arrow::array::Utf8Array;

pub(super) fn replace_lit_single_char(arr: &Utf8Array<i64>, pat: u8, val: u8) -> Utf8Array<i64> {
    let values = arr.values();
    let offsets = arr.offsets().clone();
    let validity = arr.validity().cloned();

    let mut values = values.as_slice().to_vec();
    for byte in values.iter_mut() {
        if *byte == pat {
            *byte = val;
        }
    }
    unsafe { Utf8Array::new_unchecked(arr.data_type().clone(), offsets, values.into(), validity) }
}
