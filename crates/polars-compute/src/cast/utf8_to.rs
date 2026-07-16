use arrow::array::*;
use arrow::datatypes::ArrowDataType;
use arrow::offset::Offset;
use polars_buffer::Buffer;
use polars_error::PolarsResult;
use polars_utils::unitvec;

pub(super) const RFC3339: &str = "%Y-%m-%dT%H:%M:%S%.f%:z";

pub(super) fn utf8_to_dictionary_dyn<O: Offset, K: DictionaryKey>(
    from: &dyn Array,
    ordered: bool,
) -> PolarsResult<Box<dyn Array>> {
    let values = from.as_any().downcast_ref().unwrap();
    utf8_to_dictionary::<O, K>(values, ordered).map(|x| Box::new(x) as Box<dyn Array>)
}

/// Cast [`Utf8Array`] to [`DictionaryArray`], also known as packing.
/// # Errors
/// This function errors if the maximum key is smaller than the number of distinct elements
/// in the array.
pub fn utf8_to_dictionary<O: Offset, K: DictionaryKey>(
    from: &Utf8Array<O>,
    ordered: bool,
) -> PolarsResult<DictionaryArray<K>> {
    let mut array = MutableDictionaryArray::<K, MutableUtf8Array<O>>::empty_with_value_dtype(
        from.dtype().clone(),
        ordered,
    );
    array.reserve(from.len());
    array.try_extend(from.iter())?;

    Ok(array.into())
}

/// Conversion of utf8
pub fn utf8_to_large_utf8(from: &Utf8Array<i32>) -> Utf8Array<i64> {
    let dtype = Utf8Array::<i64>::default_dtype();
    let validity = from.validity().cloned();
    let values = from.values().clone();

    let offsets = from.offsets().into();
    // SAFETY: sound because `values` fulfills the same invariants as `from.values()`
    unsafe { Utf8Array::<i64>::new_unchecked(dtype, offsets, values, validity) }
}

/// Conversion of utf8
pub fn utf8_large_to_utf8(from: &Utf8Array<i64>) -> PolarsResult<Utf8Array<i32>> {
    let dtype = Utf8Array::<i32>::default_dtype();
    let validity = from.validity().cloned();
    let values = from.values().clone();
    let offsets = from.offsets().try_into()?;

    // SAFETY: sound because `values` fulfills the same invariants as `from.values()`
    Ok(unsafe { Utf8Array::<i32>::new_unchecked(dtype, offsets, values, validity) })
}

/// Conversion to binary
pub fn utf8_to_binary<O: Offset>(from: &Utf8Array<O>, to_dtype: ArrowDataType) -> BinaryArray<O> {
    // SAFETY: erasure of an invariant is always safe
    BinaryArray::<O>::new(
        to_dtype,
        from.offsets().clone(),
        from.values().clone(),
        from.validity().cloned(),
    )
}

const ARROW_MAX_OFFSET: u32 = if cfg!(test) {
    // Used to test buffer splitting.
    i8::MAX as u32
} else {
    // Limit to i32 rather than u32 to maintain interop compatibility with arrow consumers that
    // use signed integers.
    i32::MAX as u32
};

pub fn binary_to_binview<O: Offset>(arr: &BinaryArray<O>) -> BinaryViewArray {
    let mut views = Vec::with_capacity(arr.len());

    let mut buffers = unitvec![];
    let mut current_buffer_range: std::ops::Range<usize> = 0..0;
    let mut total_buffer_len: usize = 0;

    for row_byte_range in arr
        .offsets()
        .array_windows::<2>()
        .map(|[start, end]| start.to_usize()..end.to_usize())
    {
        let row_byte_len: usize = row_byte_range.len();
        assert!(
            row_byte_len <= BINVIEW_MAX_ROW_BYTE_LEN,
            "max string/binary length exceeded"
        );

        let row_byte_values = unsafe { arr.values().get_unchecked(row_byte_range.clone()) };

        let view = if row_byte_len <= 12 {
            unsafe { View::new_inline_unchecked(row_byte_values) }
        } else {
            if row_byte_range.end > current_buffer_range.end {
                let new_buffer_end = usize::min(
                    arr.values().len(),
                    usize::max(
                        row_byte_range.end,
                        row_byte_range
                            .start
                            .saturating_add(ARROW_MAX_OFFSET as usize),
                    ),
                );

                let new_buffer_range = row_byte_range.start..new_buffer_end;

                assert!(
                    buffers.len() < u32::MAX as usize,
                    "max string/binary buffers exceeded"
                );
                buffers.push(unsafe {
                    arr.values()
                        .clone()
                        .sliced_unchecked(new_buffer_range.clone())
                });
                total_buffer_len = total_buffer_len
                    .checked_add(new_buffer_range.len())
                    .unwrap();
                current_buffer_range = new_buffer_range;
            }

            let offset: usize = row_byte_range.start - current_buffer_range.start;
            let offset: u32 = offset as u32;
            unsafe {
                View::new_noninline_unchecked(row_byte_values, (buffers.len() - 1) as u32, offset)
            }
        };

        views.push(view);
    }

    unsafe {
        BinaryViewArray::new_unchecked_unknown_md(
            ArrowDataType::BinaryView,
            views.into(),
            Buffer::from_owner(buffers),
            arr.validity().cloned(),
            Some(total_buffer_len),
        )
    }
}

pub fn utf8_to_utf8view<O: Offset>(arr: &Utf8Array<O>) -> Utf8ViewArray {
    unsafe { binary_to_binview(&arr.to_binary()).to_utf8view_unchecked() }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_overflowing_utf8_to_binview() {
        let values = [
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdaj", // 0 (offset)
            "123",                                                          // inline
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdaj", // 60
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdaj", // 0 (new buffer)
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdaj", // 60
            "234",                                                          // inline
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdaj", // 0 (new buffer)
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdaj", // 60
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdaj", // 0 (new buffer)
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdaj", // 60
            "324",                                                          // inline
        ];
        let array = Utf8Array::<i64>::from_slice(values);

        let out = utf8_to_utf8view(&array);
        // Ensure we hit the multiple buffers part.
        assert_eq!(out.data_buffers().len(), 4);
        // Ensure we created a valid binview
        let out = out.values_iter().collect::<Vec<_>>();
        assert_eq!(out, values);
    }
}
