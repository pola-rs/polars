use crate::prelude::*;
use arrow::array::{Array, ArrayData, ArrayRef, BooleanArray, PrimitiveArray};
use arrow::bitmap::Bitmap;
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowNumericType;
use arrow::error::Result as ArrowResult;
use std::ops::BitOr;
use std::sync::Arc;

pub(super) fn apply_bin_op_to_option_bitmap<F>(
    left: &Option<Bitmap>,
    right: &Option<Bitmap>,
    op: F,
) -> Result<Option<Bitmap>>
where
    F: Fn(&Bitmap, &Bitmap) -> ArrowResult<Bitmap>,
{
    match *left {
        None => match *right {
            None => Ok(None),
            Some(ref r) => Ok(Some(r.clone())),
        },
        Some(ref l) => match *right {
            None => Ok(Some(l.clone())),
            Some(ref r) => Ok(Some(op(&l, &r)?)),
        },
    }
}

/// Cache optimal zip version
pub fn zip<T>(mask: &BooleanArray, a: &PrimitiveArray<T>, b: &PrimitiveArray<T>) -> Result<ArrayRef>
where
    T: ArrowNumericType,
{
    // get data buffers
    let data_a = a.data();
    let data_b = b.data();
    let data_mask = mask.data();

    // get null bitmasks
    let mask_bitmap = data_mask.null_bitmap();
    let a_bitmap = data_a.null_bitmap();
    let b_bitmap = data_b.null_bitmap();

    // Compute final null values by bitor ops
    let bitmap = apply_bin_op_to_option_bitmap(mask_bitmap, a_bitmap, |a, b| a.bitor(b))?;
    let bitmap = apply_bin_op_to_option_bitmap(&bitmap, b_bitmap, |a, b| a.bitor(b))?;
    let null_bit_buffer = bitmap.map(|bitmap| bitmap.into_buffer());

    // Create an aligned vector.
    let mut values = AlignedVec::with_capacity_aligned(mask.len());

    // Get a slice to the values in the arrow arrays with the right offset
    let vals_a = a.value_slice(a.offset(), a.len());
    let vals_b = a.value_slice(b.offset(), b.len());

    // fill the aligned vector
    for i in 0..mask.len() {
        let take_a = mask.value(i);
        if take_a {
            unsafe {
                values.push(vals_a.get_unchecked(i));
            }
        } else {
            unsafe {
                values.push(vals_b.get_unchecked(i));
            }
        }
    }

    // give ownership to apache arrow without copying
    let (ptr, len, cap) = values.into_raw_parts();

    let buf = unsafe { Buffer::from_raw_parts(ptr as *const u8, len, cap) };
    let data = ArrayData::new(
        T::get_data_type(),
        a.len(),
        None,
        null_bit_buffer,
        a.offset(),
        vec![buf],
        vec![],
    );
    Ok(Arc::new(PrimitiveArray::<T>::from(Arc::new(data))))
}
