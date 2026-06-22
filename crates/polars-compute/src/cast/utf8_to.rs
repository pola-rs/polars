use arrow::array::*;
use arrow::datatypes::ArrowDataType;
use arrow::offset::Offset;
use arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_error::PolarsResult;
use polars_utils::vec::PushUnchecked;

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

// If we don't do this the GC of binview will trigger. As we will split up buffers into multiple
// chunks so that we don't overflow the offset u32.
fn truncate_buffer(buf: &Buffer<u8>) -> Buffer<u8> {
    // * 2, as it must be able to hold u32::MAX offset + u32::MAX len.
    let len = std::cmp::min(buf.len(), ((u32::MAX as u64) * 2) as usize);
    buf.clone().sliced(..len)
}

pub fn binary_to_binview<O: Offset>(arr: &BinaryArray<O>) -> BinaryViewArray {
    let mut views = Vec::with_capacity(arr.len());
    let mut uses_buffer = false;

    let mut base_buffer = arr.values().clone();
    // Offset into the buffer
    let mut base_ptr = base_buffer.as_ptr() as usize;

    // Offset into the binview buffers
    let mut buffer_idx = 0_u32;

    // Binview buffers
    // Note that the buffer may look far further than u32::MAX, but as we don't clone data
    let mut buffers = vec![truncate_buffer(&base_buffer)];

    for bytes in arr.values_iter() {
        let len: u32 = bytes
            .len()
            .try_into()
            .expect("max string/binary length exceeded");

        let mut payload = [0; 16];
        payload[0..4].copy_from_slice(&len.to_le_bytes());

        if len <= 12 {
            payload[4..4 + bytes.len()].copy_from_slice(bytes);
        } else {
            uses_buffer = true;

            // Copy the parts we know are correct.
            unsafe { payload[4..8].copy_from_slice(bytes.get_unchecked(0..4)) };
            payload[0..4].copy_from_slice(&len.to_le_bytes());

            let current_bytes_ptr = bytes.as_ptr() as usize;
            let offset = current_bytes_ptr - base_ptr;

            // * (offset + length) must be less than i32::MAX for compatibility with other arrow consumers.
            if let Ok(offset) = u32::try_from(offset)
                && (offset.saturating_add(len) <= ARROW_MAX_OFFSET)
            {
                #[allow(clippy::unnecessary_cast)]
                let offset = offset as u32;
                payload[12..16].copy_from_slice(&offset.to_le_bytes());
                payload[8..12].copy_from_slice(&buffer_idx.to_le_bytes());
            } else {
                let len = base_buffer.len() - offset;

                // Set new buffer
                base_buffer = base_buffer.clone().sliced(offset..offset + len);
                base_ptr = base_buffer.as_ptr() as usize;

                // And add the (truncated) one to the buffers
                buffers.push(truncate_buffer(&base_buffer));
                buffer_idx = buffer_idx.checked_add(1).expect("max buffers exceeded");

                let offset = 0u32;
                payload[12..16].copy_from_slice(&offset.to_le_bytes());
                payload[8..12].copy_from_slice(&buffer_idx.to_le_bytes());
            }
        }

        let value = View::from_le_bytes(payload);
        unsafe { views.push_unchecked(value) };
    }
    let buffers = if uses_buffer {
        Buffer::from(buffers)
    } else {
        Buffer::new()
    };
    unsafe {
        BinaryViewArray::new_unchecked_unknown_md(
            ArrowDataType::BinaryView,
            views.into(),
            buffers,
            arr.validity().cloned(),
            None,
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
