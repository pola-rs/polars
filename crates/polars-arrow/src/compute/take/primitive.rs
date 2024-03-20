use polars_utils::index::NullCount;
use polars_utils::slice::GetSaferUnchecked;

use crate::array::PrimitiveArray;
use crate::bitmap::utils::set_bit_unchecked;
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::legacy::index::IdxArr;
use crate::legacy::utils::CustomIterTools;
use crate::types::NativeType;

pub(super) unsafe fn take_values_and_validity_unchecked<T: NativeType>(
    values: &[T],
    validity_values: Option<&Bitmap>,
    indices: &IdxArr,
) -> (Vec<T>, Option<Bitmap>) {
    let index_values = indices.values().as_slice();

    let null_count = validity_values.map(|b| b.unset_bits()).unwrap_or(0);

    // first take the values, these are always needed
    let values: Vec<T> = if indices.null_count() == 0 {
        index_values
            .iter()
            .map(|idx| *values.get_unchecked_release(*idx as usize))
            .collect_trusted()
    } else {
        indices
            .iter()
            .map(|idx| match idx {
                Some(idx) => *values.get_unchecked_release(*idx as usize),
                None => T::default(),
            })
            .collect_trusted()
    };

    if null_count > 0 {
        let validity_values = validity_values.unwrap();
        // the validity buffer we will fill with all valid. And we unset the ones that are null
        // in later checks
        // this is in the assumption that most values will be valid.
        // Maybe we could add another branch based on the null count
        let mut validity = MutableBitmap::with_capacity(indices.len());
        validity.extend_constant(indices.len(), true);
        let validity_slice = validity.as_mut_slice();

        if let Some(validity_indices) = indices.validity().as_ref() {
            index_values.iter().enumerate().for_each(|(i, idx)| {
                // i is iteration count
                // idx is the index that we take from the values array.
                let idx = *idx as usize;
                if !validity_indices.get_bit_unchecked(i) || !validity_values.get_bit_unchecked(idx)
                {
                    set_bit_unchecked(validity_slice, i, false);
                }
            });
        } else {
            index_values.iter().enumerate().for_each(|(i, idx)| {
                let idx = *idx as usize;
                if !validity_values.get_bit_unchecked(idx) {
                    set_bit_unchecked(validity_slice, i, false);
                }
            });
        };
        (values, Some(validity.freeze()))
    } else {
        (values, indices.validity().cloned())
    }
}

/// Take kernel for single chunk with nulls and arrow array as index that may have nulls.
/// # Safety
/// caller must ensure indices are in bounds
pub unsafe fn take_primitive_unchecked<T: NativeType>(
    arr: &PrimitiveArray<T>,
    indices: &IdxArr,
) -> PrimitiveArray<T> {
    let (values, validity) =
        take_values_and_validity_unchecked(arr.values(), arr.validity(), indices);
    PrimitiveArray::new_unchecked(arr.data_type().clone(), values.into(), validity)
}
