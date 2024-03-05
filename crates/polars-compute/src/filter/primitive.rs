use arrow::bitmap::Bitmap;
use bytemuck::{cast_vec, cast_slice, Pod};

use super::boolean::filter_boolean_kernel;

pub fn filter_values<T: Pod>(values: &[T], mask: &Bitmap) -> Vec<T> {
    match (std::mem::size_of::<T>(), std::mem::align_of::<T>()) {
        (1, 1) => cast_vec(filter_values_u8(cast_slice(values), mask)),
        (2, 2) => cast_vec(filter_values_u16(cast_slice(values), mask)),
        (4, 4) => cast_vec(filter_values_u32(cast_slice(values), mask)),
        (8, 8) => cast_vec(filter_values_u64(cast_slice(values), mask)),
        _ => filter_values_generic(values, mask)
    }
}

fn filter_values_u8(values: &[u8], mask: &Bitmap) -> Vec<u8> {
    // TODO: fast SIMD implementation.
    filter_values_generic(values, mask)
}

fn filter_values_u16(values: &[u16], mask: &Bitmap) -> Vec<u16> {
    // TODO: fast SIMD implementation.
    filter_values_generic(values, mask)
}

fn filter_values_u32(values: &[u32], mask: &Bitmap) -> Vec<u32> {
    // TODO: fast SIMD implementation.
    filter_values_generic(values, mask)
}

fn filter_values_u64(values: &[u64], mask: &Bitmap) -> Vec<u64> {
    // TODO: fast SIMD implementation.
    filter_values_generic(values, mask)
}

fn filter_values_generic<T: Pod>(values: &[T], mask: &Bitmap) -> Vec<T> {
    assert_eq!(values.len(), mask.len());
    let mask_bits_set = mask.set_bits();
    let mut out = Vec::with_capacity(mask_bits_set + 1);
    unsafe {
        super::scalar::filter_scalar_values(values, mask, out.as_mut_ptr());
        out.set_len(mask_bits_set);
    }
    out
}

pub fn filter_values_and_validity<T: Pod>(
    values: &[T],
    validity: Option<&Bitmap>,
    mask: &Bitmap,
) -> (Vec<T>, Option<Bitmap>) {
    (
        filter_values(values, mask),
        validity.map(|v| filter_boolean_kernel(v, mask)),
    )
}
