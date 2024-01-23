use super::Index;
use crate::array::{Array, BooleanArray, PrimitiveArray};
use crate::bitmap::{Bitmap, MutableBitmap};

// take implementation when neither values nor indices contain nulls
unsafe fn take_no_validity<I: Index>(values: &Bitmap, indices: &[I]) -> (Bitmap, Option<Bitmap>) {
    let values = indices
        .iter()
        .map(|index| values.get_bit_unchecked(index.to_usize()));
    let buffer = Bitmap::from_trusted_len_iter(values);

    (buffer, None)
}

// take implementation when only values contain nulls
unsafe fn take_values_validity<I: Index>(
    values: &BooleanArray,
    indices: &[I],
) -> (Bitmap, Option<Bitmap>) {
    let validity_values = values.validity().unwrap();
    let validity = indices
        .iter()
        .map(|index| validity_values.get_bit_unchecked(index.to_usize()));
    let validity = Bitmap::from_trusted_len_iter(validity);

    let values_values = values.values();
    let values = indices
        .iter()
        .map(|index| values_values.get_bit_unchecked(index.to_usize()));
    let buffer = Bitmap::from_trusted_len_iter(values);

    (buffer, validity.into())
}

// take implementation when only indices contain nulls
pub(super) unsafe fn take_indices_validity<I: Index>(
    values: &Bitmap,
    indices: &PrimitiveArray<I>,
) -> (Bitmap, Option<Bitmap>) {
    let validity = indices.validity().unwrap();

    let values = indices.values().iter().enumerate().map(|(i, index)| {
        let index = index.to_usize();
        match values.get(index) {
            Some(value) => value,
            None => validity.get_bit_unchecked(i),
        }
    });

    let buffer = Bitmap::from_trusted_len_iter(values);

    (buffer, indices.validity().cloned())
}

// take implementation when both values and indices contain nulls
unsafe fn take_values_indices_validity<I: Index>(
    values: &BooleanArray,
    indices: &PrimitiveArray<I>,
) -> (Bitmap, Option<Bitmap>) {
    let mut validity = MutableBitmap::with_capacity(indices.len());

    let values_validity = values.validity().unwrap();

    let values_values = values.values();
    let values = indices.iter().map(|index| match index {
        Some(index) => {
            let index = index.to_usize();
            validity.push(values_validity.get_bit_unchecked(index));
            values_values.get_bit_unchecked(index)
        },
        None => {
            validity.push(false);
            false
        },
    });
    let values = Bitmap::from_trusted_len_iter(values);
    (values, validity.into())
}

/// `take` implementation for boolean arrays
pub(super) unsafe fn take_unchecked<I: Index>(
    values: &BooleanArray,
    indices: &PrimitiveArray<I>,
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
