use arrow::bitmap::Bitmap;
use arrow::buffer::Buffer;
use arrow::error::Result;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
use arrow::{
    array::{ArrayDataRef, PrimitiveArray},
    datatypes::ArrowNumericType,
};
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
use num::One;

/// Performs a SIMD load but sets all 'invalid' lanes to a constant value.
///
/// 'invalid' lanes are lanes where the corresponding array slots are either `NULL` or between the
/// length and capacity of the array, i.e. in the padded region.
///
/// Note that `array` below has it's own `Bitmap` separate from the `bitmap` argument.  This
/// function is used to prepare `array`'s for binary operations.  The `bitmap` argument is the
/// `Bitmap` after the binary operation.
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
pub(crate) unsafe fn simd_load_set_invalid<T>(
    array: &PrimitiveArray<T>,
    bitmap: &Option<Bitmap>,
    i: usize,
    simd_width: usize,
    fill_value: T::Native,
) -> T::Simd
where
    T: ArrowNumericType,
    T::Native: One,
{
    let simd_with_zeros = T::load(array.value_slice(i, simd_width));
    T::mask_select(
        is_valid::<T>(bitmap, i, simd_width, array.len()),
        simd_with_zeros,
        T::init(fill_value),
    )
}

/// Applies a given binary operation, `op`, to two references to `Option<Bitmap>`'s.
///
/// This function is useful when implementing operations on higher level arrays.
pub(super) fn apply_bin_op_to_option_bitmap<F>(
    left: &Option<Bitmap>,
    right: &Option<Bitmap>,
    op: F,
) -> Result<Option<Buffer>>
where
    F: Fn(&Buffer, &Buffer) -> Result<Buffer>,
{
    match *left {
        None => match *right {
            None => Ok(None),
            Some(ref r) => Ok(Some(r.buffer_ref().clone())),
        },
        Some(ref l) => match *right {
            None => Ok(Some(l.buffer_ref().clone())),
            Some(ref r) => Ok(Some(op(&l.buffer_ref(), &r.buffer_ref())?)),
        },
    }
}
