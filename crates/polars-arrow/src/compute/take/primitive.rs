use super::Index;
use crate::array::{Array, PrimitiveArray};
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::buffer::Buffer;
use crate::types::NativeType;

// take implementation when neither values nor indices contain nulls
unsafe fn take_no_validity<T: NativeType, I: Index>(
    values: &[T],
    indices: &[I],
) -> (Buffer<T>, Option<Bitmap>) {
    let values = indices
        .iter()
        .map(|index| *values.get_unchecked(index.to_usize()))
        .collect::<Vec<_>>();

    (values.into(), None)
}

// take implementation when only values contain nulls
unsafe fn take_values_validity<T: NativeType, I: Index>(
    values: &PrimitiveArray<T>,
    indices: &[I],
) -> (Buffer<T>, Option<Bitmap>) {
    let values_validity = values.validity().unwrap();

    let validity = indices
        .iter()
        .map(|index| values_validity.get_bit_unchecked(index.to_usize()));
    let validity = MutableBitmap::from_trusted_len_iter(validity);

    let values_values = values.values();

    let values = indices
        .iter()
        .map(|index| *values_values.get_unchecked(index.to_usize()))
        .collect::<Vec<_>>();

    (values.into(), validity.into())
}

// take implementation when only indices contain nulls
unsafe fn take_indices_validity<T: NativeType, I: Index>(
    values: &[T],
    indices: &PrimitiveArray<I>,
) -> (Buffer<T>, Option<Bitmap>) {
    let values = indices
        .values()
        .iter()
        .map(|index| {
            let index = index.to_usize();
            *values.get_unchecked(index)
        })
        .collect::<Vec<_>>();

    (values.into(), indices.validity().cloned())
}

// take implementation when both values and indices contain nulls
unsafe fn take_values_indices_validity<T: NativeType, I: Index>(
    values: &PrimitiveArray<T>,
    indices: &PrimitiveArray<I>,
) -> (Buffer<T>, Option<Bitmap>) {
    let mut bitmap = MutableBitmap::with_capacity(indices.len());

    let values_validity = values.validity().unwrap();

    let values_values = values.values();
    let values = indices
        .iter()
        .map(|index| match index {
            Some(index) => {
                let index = index.to_usize();
                bitmap.push(values_validity.get_bit_unchecked(index));
                *values_values.get_unchecked(index)
            },
            None => {
                bitmap.push(false);
                T::default()
            },
        })
        .collect::<Vec<_>>();
    (values.into(), bitmap.into())
}

/// `take` implementation for primitive arrays
pub(super) unsafe fn take_unchecked<T: NativeType, I: Index>(
    values: &PrimitiveArray<T>,
    indices: &PrimitiveArray<I>,
) -> PrimitiveArray<T> {
    let indices_has_validity = indices.null_count() > 0;
    let values_has_validity = values.null_count() > 0;
    let (buffer, validity) = match (values_has_validity, indices_has_validity) {
        (false, false) => take_no_validity::<T, I>(values.values(), indices.values()),
        (true, false) => take_values_validity::<T, I>(values, indices.values()),
        (false, true) => take_indices_validity::<T, I>(values.values(), indices),
        (true, true) => take_values_indices_validity::<T, I>(values, indices),
    };

    PrimitiveArray::<T>::new(values.data_type().clone(), buffer, validity)
}
