use crate::error::{PolarsError, Result};
use arrow::array::{Array, ArrayData, PrimitiveArray, UInt32Array};
use arrow::buffer::{Buffer, MutableBuffer};
use arrow::datatypes::{ArrowNativeType, ArrowPrimitiveType};
use arrow::util::bit_util;
use std::sync::Arc;

pub fn set_at_idx<T>(
    array: &PrimitiveArray<T>,
    idx: UInt32Array,
    set_value: T::Native,
) -> Result<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    T::Native: ArrowNativeType,
{
    let data = array.data_ref();

    // Clone the data.
    let new_buf = unsafe { Buffer::from_trusted_len_iter(array.values().into_iter().map(|v| *v)) };

    // Create a writable slice from the buffer
    let new_buf_data = unsafe {
        let ptr = new_buf.as_ptr() as *mut T::Native;
        std::slice::from_raw_parts_mut(ptr, array.len())
    };

    let mut null_bits = MutableBuffer::new(0);

    if let Some(buf) = data.null_buffer() {
        null_bits.extend_from_slice(buf.as_slice())
    } else {
        // set all values to valid
        let num_bytes = bit_util::ceil(array.len(), 8);
        null_bits.extend((0..num_bytes).map(|_| u8::MAX))
    }
    let null_bits_data = null_bits.as_mut();

    idx.into_iter().try_for_each::<_, Result<_>>(|opt_idx| {
        if let Some(idx) = opt_idx {
            let idx = idx as usize;
            let value = new_buf_data
                .get_mut(idx)
                .ok_or_else(|| PolarsError::Other("out of bounds".into()))?;
            *value = set_value;
            bit_util::set_bit(null_bits_data, idx);
        }
        Ok(())
    })?;

    let null_bit_buffer: Buffer = null_bits.into();
    let len = array.len();
    let null_count = Some(len - null_bit_buffer.count_set_bits());

    let data = ArrayData::new(
        T::DATA_TYPE,
        len,
        null_count,
        Some(null_bit_buffer),
        0,
        vec![new_buf],
        vec![],
    );
    Ok(PrimitiveArray::from(Arc::new(data)))
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::array::{Int32Array, UInt32Array};

    #[test]
    fn test_set_kernel() {
        let idx = UInt32Array::from(vec![Some(1), None, Some(2)]);
        let values = Int32Array::from(vec![Some(1), None, None]);
        let out = set_at_idx(&values, idx, 4).unwrap();
        dbg!(out);
    }
}
