use arrow::array::{Array, BooleanArray, PrimitiveArray};
use arrow::bitmap::{Bitmap, MutableBitmap};

use crate::index::IdxSize;

unsafe fn take_values(values: &Bitmap, indices: &[IdxSize]) -> Bitmap {
    let values = indices.iter().map(|&index| {
        debug_assert!((index as usize) < values.len());
        values.get_bit_unchecked(index as usize)
    });
    Bitmap::from_trusted_len_iter(values)
}

// take implementation when neither values nor indices contain nulls
unsafe fn take_no_validity(values: &Bitmap, indices: &[IdxSize]) -> (Bitmap, Option<Bitmap>) {
    (take_values(values, indices), None)
}

// take implementation when only values contain nulls
unsafe fn take_values_validity(
    values: &BooleanArray,
    indices: &[IdxSize],
) -> (Bitmap, Option<Bitmap>) {
    let validity_values = values.validity().unwrap();
    let validity = take_values(validity_values, indices);

    let values_values = values.values();
    let buffer = take_values(values_values, indices);

    (buffer, validity.into())
}

// take implementation when only indices contain nulls
unsafe fn take_indices_validity(
    values: &Bitmap,
    indices: &PrimitiveArray<IdxSize>,
) -> (Bitmap, Option<Bitmap>) {
    // simply take all and copy the bitmap
    let buffer = take_values(values, indices.values());

    (buffer, indices.validity().cloned())
}

// take implementation when both values and indices contain nulls
unsafe fn take_values_indices_validity(
    values: &BooleanArray,
    indices: &PrimitiveArray<IdxSize>,
) -> (Bitmap, Option<Bitmap>) {
    let mut validity = MutableBitmap::with_capacity(indices.len());

    let values_validity = values.validity().unwrap();

    let values_values = values.values();
    let values = indices.iter().map(|index| match index {
        Some(&index) => {
            let index = index as usize;
            debug_assert!(index < values.len());
            validity.push(values_validity.get_bit_unchecked(index));
            values_values.get_bit_unchecked(index)
        }
        None => {
            validity.push(false);
            false
        }
    });
    let values = Bitmap::from_trusted_len_iter(values);
    (values, validity.into())
}

/// `take` implementation for boolean arrays
pub unsafe fn take_unchecked(
    values: &BooleanArray,
    indices: &PrimitiveArray<IdxSize>,
) -> BooleanArray {
    let data_type = values.data_type().clone();
    let indices_has_validity = indices.null_count() > 0;
    let values_has_validity = values.null_count() > 0;

    let (values, validity) = match (values_has_validity, indices_has_validity) {
        (false, false) => take_no_validity(values.values(), indices.values()),
        (true, false) => take_values_validity(values, indices.values()),
        (false, true) => take_indices_validity(values.values(), indices),
        (true, true) => take_values_indices_validity(values, indices),
    };

    BooleanArray::new(data_type, values, validity)
}
