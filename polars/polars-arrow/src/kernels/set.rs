use crate::builder::PrimitiveArrayBuilder;
use crate::kernels::BinaryMaskedSliceIterator;
use crate::vec::AlignedVec;
use arrow::array::*;
use arrow::datatypes::{ArrowNativeType, ArrowNumericType};

/// Set values in a primitive array based on a mask array. This is fast when large chunks of bits are set or unset.
pub fn set_with_mask<T>(
    array: &PrimitiveArray<T>,
    mask: &BooleanArray,
    value: T::Native,
) -> PrimitiveArray<T>
where
    T: ArrowNumericType,
    T::Native: ArrowNativeType,
{
    debug_assert!(mask.null_count() == 0);
    let values = array.values();

    if array.null_count() == 0 {
        let mut av = AlignedVec::with_capacity_aligned(array.len());
        BinaryMaskedSliceIterator::new(mask)
            .into_iter()
            .for_each(|(lower, upper, truthy)| {
                if truthy {
                    av.extend((lower..upper).map(|_| value))
                } else {
                    av.extend_from_slice(&values[lower..upper])
                }
            });
        av.into_primitive_array(None)
    } else {
        let mask_values = &mask.data_ref().buffers()[0];

        // this operation is performed before iteration
        // because it is fast and allows reserving all the needed memory
        let pop_count = mask_values.count_set_bits_offset(mask.offset(), mask.len());

        let mut builder = PrimitiveArrayBuilder::new(pop_count);
        BinaryMaskedSliceIterator::new(mask)
            .into_iter()
            .for_each(|(lower, upper, truthy)| {
                if truthy {
                    for _ in lower..upper {
                        builder.append_value(value)
                    }
                } else {
                    for idx in lower..upper {
                        if array.is_valid(idx) {
                            // Safety
                            // idx is within bounds
                            builder.append_value(unsafe { *values.get_unchecked(idx) })
                        } else {
                            builder.append_null()
                        }
                    }
                }
            });

        builder.finish()
    }
}

#[cfg(feature = "future")]
pub fn set_at_idx<T>(
    array: &PrimitiveArray<T>,
    idx: UInt32Array,
    set_value: T::Native,
) -> Result<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    T::Native: ArrowNativeType,
{
    use crate::error::{PolarsError, Result};
    use arrow::buffer::{Buffer, MutableBuffer};
    use arrow::util::bit_util;
    use std::sync::Arc;
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
    use arrow::array::UInt32Array;

    #[test]
    fn test_set_mask() {
        let mask = BooleanArray::from((0..86).map(|v| v > 68 && v != 85).collect::<Vec<bool>>());
        let val = UInt32Array::from((0..86).collect::<Vec<_>>());
        let a = set_with_mask(&val, &mask, 100);
        let slice = a.values();

        dbg!(&slice, slice.len());
        assert_eq!(slice[a.len() - 1], 85);
        assert_eq!(slice[a.len() - 2], 100);
        assert_eq!(slice[67], 67);
        assert_eq!(slice[68], 68);
        assert_eq!(slice[1], 1);
        assert_eq!(slice[0], 0);
    }
}
