use arrow::array::{Array, MutableBinaryViewArray, Utf8ViewArray};

/// Returns a Utf8Array<O> with a substring starting from `start` and with optional length `length` of each of the elements in `array`.
/// `start` can be negative, in which case the start counts from the end of the string.
pub(super) fn utf8_substring(
    array: &Utf8ViewArray,
    start: i64,
    length: &Option<u64>,
) -> Utf8ViewArray {
    let length = length.map(|v| v as usize);

    let iter = array.values_iter().map(|str_val| {
        // compute where we should start slicing this entry.
        let start = if start >= 0 {
            start as usize
        } else {
            let start = (0i64 - start) as usize;
            str_val
                .char_indices()
                .rev()
                .nth(start)
                .map(|(idx, _)| idx + 1)
                .unwrap_or(0)
        };

        let mut iter_chars = str_val.char_indices();
        if let Some((start_idx, _)) = iter_chars.nth(start) {
            // length of the str
            let len_end = str_val.len() - start_idx;

            // length to slice
            let length = length.unwrap_or(len_end);

            if length == 0 {
                return "";
            }
            // compute
            let end_idx = iter_chars
                .nth(length.saturating_sub(1))
                .map(|(idx, _)| idx)
                .unwrap_or(str_val.len());

            &str_val[start_idx..end_idx]
        } else {
            ""
        }
    });

    MutableBinaryViewArray::from_values_iter(iter)
        .freeze()
        .with_validity(array.validity().cloned())
}
