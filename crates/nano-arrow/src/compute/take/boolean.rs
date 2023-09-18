use crate::{
    array::{Array, BooleanArray, PrimitiveArray},
    bitmap::{Bitmap, MutableBitmap},
};

use super::Index;

// take implementation when neither values nor indices contain nulls
fn take_no_validity<I: Index>(values: &Bitmap, indices: &[I]) -> (Bitmap, Option<Bitmap>) {
    let values = indices.iter().map(|index| values.get_bit(index.to_usize()));
    let buffer = Bitmap::from_trusted_len_iter(values);

    (buffer, None)
}

// take implementation when only values contain nulls
fn take_values_validity<I: Index>(
    values: &BooleanArray,
    indices: &[I],
) -> (Bitmap, Option<Bitmap>) {
    let validity_values = values.validity().unwrap();
    let validity = indices
        .iter()
        .map(|index| validity_values.get_bit(index.to_usize()));
    let validity = Bitmap::from_trusted_len_iter(validity);

    let values_values = values.values();
    let values = indices
        .iter()
        .map(|index| values_values.get_bit(index.to_usize()));
    let buffer = Bitmap::from_trusted_len_iter(values);

    (buffer, validity.into())
}

// take implementation when only indices contain nulls
fn take_indices_validity<I: Index>(
    values: &Bitmap,
    indices: &PrimitiveArray<I>,
) -> (Bitmap, Option<Bitmap>) {
    let validity = indices.validity().unwrap();

    let values = indices.values().iter().enumerate().map(|(i, index)| {
        let index = index.to_usize();
        match values.get(index) {
            Some(value) => value,
            None => {
                if !validity.get_bit(i) {
                    false
                } else {
                    panic!("Out-of-bounds index {index}")
                }
            }
        }
    });

    let buffer = Bitmap::from_trusted_len_iter(values);

    (buffer, indices.validity().cloned())
}

// take implementation when both values and indices contain nulls
fn take_values_indices_validity<I: Index>(
    values: &BooleanArray,
    indices: &PrimitiveArray<I>,
) -> (Bitmap, Option<Bitmap>) {
    let mut validity = MutableBitmap::with_capacity(indices.len());

    let values_validity = values.validity().unwrap();

    let values_values = values.values();
    let values = indices.iter().map(|index| match index {
        Some(index) => {
            let index = index.to_usize();
            validity.push(values_validity.get_bit(index));
            values_values.get_bit(index)
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
pub fn take<I: Index>(values: &BooleanArray, indices: &PrimitiveArray<I>) -> BooleanArray {
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

#[cfg(test)]
mod tests {
    use crate::array::Int32Array;

    use super::*;

    fn _all_cases() -> Vec<(Int32Array, BooleanArray, BooleanArray)> {
        vec![
            (
                Int32Array::from(&[Some(1), Some(0)]),
                BooleanArray::from(vec![Some(true), Some(false)]),
                BooleanArray::from(vec![Some(false), Some(true)]),
            ),
            (
                Int32Array::from(&[Some(1), None]),
                BooleanArray::from(vec![Some(true), Some(false)]),
                BooleanArray::from(vec![Some(false), None]),
            ),
            (
                Int32Array::from(&[Some(1), Some(0)]),
                BooleanArray::from(vec![None, Some(false)]),
                BooleanArray::from(vec![Some(false), None]),
            ),
            (
                Int32Array::from(&[Some(1), None, Some(0)]),
                BooleanArray::from(vec![None, Some(false)]),
                BooleanArray::from(vec![Some(false), None, None]),
            ),
        ]
    }

    #[test]
    fn all_cases() {
        let cases = _all_cases();
        for (indices, input, expected) in cases {
            let output = take(&input, &indices);
            assert_eq!(expected, output);
        }
    }
}
