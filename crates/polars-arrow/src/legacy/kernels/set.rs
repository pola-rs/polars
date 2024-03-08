use std::ops::BitOr;

use polars_error::polars_err;
use polars_utils::IdxSize;

use crate::array::*;
use crate::datatypes::ArrowDataType;
use crate::legacy::array::default_arrays::FromData;
use crate::legacy::error::PolarsResult;
use crate::legacy::kernels::BinaryMaskedSliceIterator;
use crate::legacy::trusted_len::TrustedLenPush;
use crate::types::NativeType;

/// Set values in a primitive array where the primitive array has null values.
/// this is faster because we don't have to invert and combine bitmaps
pub fn set_at_nulls<T>(array: &PrimitiveArray<T>, value: T) -> PrimitiveArray<T>
where
    T: NativeType,
{
    let values = array.values();
    if array.null_count() == 0 {
        return array.clone();
    }

    let validity = array.validity().unwrap();
    let validity = BooleanArray::from_data_default(validity.clone(), None);

    let mut av = Vec::with_capacity(array.len());
    BinaryMaskedSliceIterator::new(&validity).for_each(|(lower, upper, truthy)| {
        if truthy {
            av.extend_from_slice(&values[lower..upper])
        } else {
            av.extend_trusted_len(std::iter::repeat(value).take(upper - lower))
        }
    });

    PrimitiveArray::new(array.data_type().clone(), av.into(), None)
}

/// Set values in a primitive array based on a mask array. This is fast when large chunks of bits are set or unset.
pub fn set_with_mask<T: NativeType>(
    array: &PrimitiveArray<T>,
    mask: &BooleanArray,
    value: T,
    data_type: ArrowDataType,
) -> PrimitiveArray<T> {
    let values = array.values();

    let mut buf = Vec::with_capacity(array.len());
    BinaryMaskedSliceIterator::new(mask).for_each(|(lower, upper, truthy)| {
        if truthy {
            buf.extend_trusted_len(std::iter::repeat(value).take(upper - lower))
        } else {
            buf.extend_from_slice(&values[lower..upper])
        }
    });
    // make sure that where the mask is set to true, the validity buffer is also set to valid
    // after we have applied the or operation we have new buffer with no offsets
    let validity = array.validity().as_ref().map(|valid| {
        let mask_bitmap = mask.values();
        valid.bitor(mask_bitmap)
    });

    PrimitiveArray::new(data_type, buf.into(), validity)
}

/// Efficiently sets value at the indices from the iterator to `set_value`.
/// The new array is initialized with a `memcpy` from the old values.
pub fn scatter_single_non_null<T, I>(
    array: &PrimitiveArray<T>,
    idx: I,
    set_value: T,
    data_type: ArrowDataType,
) -> PolarsResult<PrimitiveArray<T>>
where
    T: NativeType,
    I: IntoIterator<Item = IdxSize>,
{
    let mut buf = Vec::with_capacity(array.len());
    buf.extend_from_slice(array.values().as_slice());
    let mut_slice = buf.as_mut_slice();

    idx.into_iter().try_for_each::<_, PolarsResult<_>>(|idx| {
        let val = mut_slice
            .get_mut(idx as usize)
            .ok_or_else(|| polars_err!(ComputeError: "index is out of bounds"))?;
        *val = set_value;
        Ok(())
    })?;

    Ok(PrimitiveArray::new(
        data_type,
        buf.into(),
        array.validity().cloned(),
    ))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_set_mask() {
        let mask = BooleanArray::from_iter((0..86).map(|v| v > 68 && v != 85).map(Some));
        let val = UInt32Array::from_iter((0..86).map(Some));
        let a = set_with_mask(&val, &mask, 100, ArrowDataType::UInt32);
        let slice = a.values();

        assert_eq!(slice[a.len() - 1], 85);
        assert_eq!(slice[a.len() - 2], 100);
        assert_eq!(slice[67], 67);
        assert_eq!(slice[68], 68);
        assert_eq!(slice[1], 1);
        assert_eq!(slice[0], 0);

        let mask = BooleanArray::from_slice([
            false, true, false, true, false, true, false, true, false, false,
        ]);
        let val = UInt32Array::from_slice([0; 10]);
        let out = set_with_mask(&val, &mask, 1, ArrowDataType::UInt32);
        assert_eq!(out.values().as_slice(), &[0, 1, 0, 1, 0, 1, 0, 1, 0, 0]);

        let val = UInt32Array::from(&[None, None, None]);
        let mask = BooleanArray::from(&[Some(true), Some(true), None]);
        let out = set_with_mask(&val, &mask, 1, ArrowDataType::UInt32);
        let out: Vec<_> = out.iter().map(|v| v.copied()).collect();
        assert_eq!(out, &[Some(1), Some(1), None])
    }

    #[test]
    fn test_scatter_single_non_null() {
        let val = UInt32Array::from_slice([1, 2, 3]);
        let out =
            scatter_single_non_null(&val, std::iter::once(1), 100, ArrowDataType::UInt32).unwrap();
        assert_eq!(out.values().as_slice(), &[1, 100, 3]);
        let out = scatter_single_non_null(&val, std::iter::once(100), 100, ArrowDataType::UInt32);
        assert!(out.is_err())
    }
}
