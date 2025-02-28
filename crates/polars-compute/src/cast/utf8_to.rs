use std::sync::Arc;

use arrow::array::*;
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;
use arrow::offset::Offset;
use arrow::types::NativeType;
use polars_error::PolarsResult;
use polars_utils::vec::PushUnchecked;

pub(super) const RFC3339: &str = "%Y-%m-%dT%H:%M:%S%.f%:z";

pub(super) fn utf8_to_dictionary_dyn<O: Offset, K: DictionaryKey>(
    from: &dyn Array,
) -> PolarsResult<Box<dyn Array>> {
    let values = from.as_any().downcast_ref().unwrap();
    utf8_to_dictionary::<O, K>(values).map(|x| Box::new(x) as Box<dyn Array>)
}

/// Cast [`Utf8Array`] to [`DictionaryArray`], also known as packing.
/// # Errors
/// This function errors if the maximum key is smaller than the number of distinct elements
/// in the array.
pub fn utf8_to_dictionary<O: Offset, K: DictionaryKey>(
    from: &Utf8Array<O>,
) -> PolarsResult<DictionaryArray<K>> {
    let mut array = MutableDictionaryArray::<K, MutableUtf8Array<O>>::new();
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

// Different types to test the overflow path.
#[cfg(not(test))]
type OffsetType = u32;

// To trigger overflow
#[cfg(test)]
type OffsetType = i8;

// If we don't do this the GC of binview will trigger. As we will split up buffers into multiple
// chunks so that we don't overflow the offset u32.
fn truncate_buffer(buf: &Buffer<u8>) -> Buffer<u8> {
    // * 2, as it must be able to hold u32::MAX offset + u32::MAX len.
    buf.clone().sliced(
        0,
        std::cmp::min(buf.len(), ((OffsetType::MAX as u64) * 2) as usize),
    )
}

pub fn binary_to_binview<O: Offset>(arr: &BinaryArray<O>) -> BinaryViewArray {
    // Ensure we didn't accidentally set wrong type
    #[cfg(not(debug_assertions))]
    let _ = std::mem::transmute::<OffsetType, u32>;

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

            // Here we check the overflow of the buffer offset.
            if let Ok(offset) = OffsetType::try_from(offset) {
                #[allow(clippy::unnecessary_cast)]
                let offset = offset as u32;
                payload[12..16].copy_from_slice(&offset.to_le_bytes());
                payload[8..12].copy_from_slice(&buffer_idx.to_le_bytes());
            } else {
                let len = base_buffer.len() - offset;

                // Set new buffer
                base_buffer = base_buffer.clone().sliced(offset, len);
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
        Arc::from(buffers)
    } else {
        Arc::from([])
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
    fn overflowing_utf8_to_binview() {
        let values = [
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdajfkasjdflkasjdf", // 0 (offset)
            "123",                                                                        // inline
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdajfkasjdflkasjdf", // 74
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdajfkasjdflkasjdf", // 0 (new buffer)
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdajfkasjdflkasjdf", // 74
            "234",                                                                        // inline
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdajfkasjdflkasjdf", // 0 (new buffer)
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdajfkasjdflkasjdf", // 74
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdajfkasjdflkasjdf", // 0 (new buffer)
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdajfkasjdflkasjdf", // 74
            "324",                                                                        // inline
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
