use crate::error::{PolarsError, Result};
use crate::kernels::BinaryMaskedSliceIterator;
use crate::utils::{buffer_or, combine_null_buffers};
use crate::vec::AlignedVec;
use arrow::array::*;
use arrow::datatypes::{ArrowNativeType, ArrowNumericType, ArrowPrimitiveType};

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
    let values = array.values();

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
    // make sure that where the mask is set to true, the validity buffer is also set to valid
    // after we have applied the or operation we have new buffer with no offsets
    let validity = array.data().null_buffer().map(|buf| {
        let mask_buf = mask.values();
        buffer_or(mask_buf, mask.offset(), buf, array.offset(), array.len())
    });

    // now we also combine it with the null buffer of the mask
    let validity = combine_null_buffers(
        validity.as_ref(),
        0,
        mask.data().null_buffer(),
        mask.offset(),
        array.len(),
    );
    av.into_primitive_array(validity)
}

/// Efficiently sets value at the indices from the iterator to `set_value`.
/// The new array is initialized with a `memcpy` from the old values.
pub fn set_at_idx_no_null<T, I>(
    array: &PrimitiveArray<T>,
    idx: I,
    set_value: T::Native,
) -> Result<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    T::Native: ArrowNativeType,
    I: IntoIterator<Item = usize>,
{
    debug_assert_eq!(array.null_count(), 0);
    let mut av = AlignedVec::new_from_slice(array.values());
    idx.into_iter().try_for_each::<_, Result<_>>(|idx| {
        let val = av
            .inner
            .get_mut(idx)
            .ok_or_else(|| PolarsError::OutOfBounds("idx is out of bounds".into()))?;
        *val = set_value;
        Ok(())
    })?;
    Ok(av.into_primitive_array(None))
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

        assert_eq!(slice[a.len() - 1], 85);
        assert_eq!(slice[a.len() - 2], 100);
        assert_eq!(slice[67], 67);
        assert_eq!(slice[68], 68);
        assert_eq!(slice[1], 1);
        assert_eq!(slice[0], 0);

        let mask = BooleanArray::from(vec![
            false, true, false, true, false, true, false, true, false, false,
        ]);
        let val = UInt32Array::from(vec![0; 10]);
        let out = set_with_mask(&val, &mask, 1);
        dbg!(&out);
        assert_eq!(out.values(), &[0, 1, 0, 1, 0, 1, 0, 1, 0, 0]);

        let val = UInt32Array::from(vec![None, None, None]);
        let mask = BooleanArray::from(vec![Some(true), Some(true), None]);
        let out = set_with_mask(&val, &mask, 1);
        let out: Vec<_> = out.iter().collect();
        assert_eq!(out, &[Some(1), Some(1), None])
    }

    #[test]
    fn test_set_at_idx() {
        let val = UInt32Array::from(vec![1, 2, 3]);
        let out = set_at_idx_no_null(&val, std::iter::once(1), 100).unwrap();
        assert_eq!(out.values(), &[1, 100, 3]);
        let out = set_at_idx_no_null(&val, std::iter::once(100), 100);
        assert!(out.is_err())
    }
}
