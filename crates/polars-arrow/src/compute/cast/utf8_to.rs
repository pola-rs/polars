use std::sync::Arc;

use polars_error::PolarsResult;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::vec::PushUnchecked;

use crate::array::*;
use crate::datatypes::ArrowDataType;
use crate::offset::Offset;
use crate::types::NativeType;

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
    array.try_extend(from.iter())?;

    Ok(array.into())
}

/// Conversion of utf8
pub fn utf8_to_large_utf8(from: &Utf8Array<i32>) -> Utf8Array<i64> {
    let data_type = Utf8Array::<i64>::default_data_type();
    let validity = from.validity().cloned();
    let values = from.values().clone();

    let offsets = from.offsets().into();
    // Safety: sound because `values` fulfills the same invariants as `from.values()`
    unsafe { Utf8Array::<i64>::new_unchecked(data_type, offsets, values, validity) }
}

/// Conversion of utf8
pub fn utf8_large_to_utf8(from: &Utf8Array<i64>) -> PolarsResult<Utf8Array<i32>> {
    let data_type = Utf8Array::<i32>::default_data_type();
    let validity = from.validity().cloned();
    let values = from.values().clone();
    let offsets = from.offsets().try_into()?;

    // Safety: sound because `values` fulfills the same invariants as `from.values()`
    Ok(unsafe { Utf8Array::<i32>::new_unchecked(data_type, offsets, values, validity) })
}

/// Conversion to binary
pub fn utf8_to_binary<O: Offset>(
    from: &Utf8Array<O>,
    to_data_type: ArrowDataType,
) -> BinaryArray<O> {
    // Safety: erasure of an invariant is always safe
    unsafe {
        BinaryArray::<O>::new(
            to_data_type,
            from.offsets().clone(),
            from.values().clone(),
            from.validity().cloned(),
        )
    }
}

pub fn binary_to_binview<O: Offset>(arr: &BinaryArray<O>) -> BinaryViewArray {
    let buffer_idx = 0_u32;
    let base_ptr = arr.values().as_ptr() as usize;

    let mut views = Vec::with_capacity(arr.len());
    let mut uses_buffer = false;
    for bytes in arr.values_iter() {
        let len: u32 = bytes.len().try_into().unwrap();

        let mut payload = [0; 16];
        payload[0..4].copy_from_slice(&len.to_le_bytes());

        if len <= 12 {
            payload[4..4 + bytes.len()].copy_from_slice(bytes);
        } else {
            uses_buffer = true;
            unsafe { payload[4..8].copy_from_slice(bytes.get_unchecked_release(0..4)) };
            let offset = (bytes.as_ptr() as usize - base_ptr) as u32;
            payload[0..4].copy_from_slice(&len.to_le_bytes());
            payload[8..12].copy_from_slice(&buffer_idx.to_le_bytes());
            payload[12..16].copy_from_slice(&offset.to_le_bytes());
        }

        let value = View::from_le_bytes(payload);
        unsafe { views.push_unchecked(value) };
    }
    let buffers = if uses_buffer {
        Arc::from([arr.values().clone()])
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
