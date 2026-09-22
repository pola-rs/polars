use bytemuck::{Pod, cast_slice};
use polars_arrow::bitmap::Bitmap;
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use polars_utils::cpuid::is_avx512_enabled;

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use super::avx512;
use super::boolean::filter_boolean_kernel;
use super::scalar::{scalar_filter, scalar_filter_offset};

type FilterFn<T> = for<'a> unsafe fn(&'a [T], &'a [u8], *mut T) -> (&'a [T], &'a [u8], *mut T);

fn nop_filter<'a, T: Pod>(
    values: &'a [T],
    mask: &'a [u8],
    out: *mut T,
) -> (&'a [T], &'a [u8], *mut T) {
    (values, mask, out)
}

/// A bulk kernel and the number of elements it may write past the ones it selects.
type Bulk<T> = (usize, FilterFn<T>);

fn bulk_u8() -> Bulk<u8> {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    if is_avx512_enabled() && std::arch::is_x86_feature_detected!("avx512vbmi2") {
        return (64, avx512::filter_u8_avx512vbmi2);
    }

    (1, nop_filter)
}

fn bulk_u16() -> Bulk<u16> {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    if is_avx512_enabled() && std::arch::is_x86_feature_detected!("avx512vbmi2") {
        return (32, avx512::filter_u16_avx512vbmi2);
    }

    (1, nop_filter)
}

fn bulk_u32() -> Bulk<u32> {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    if is_avx512_enabled() {
        return (16, avx512::filter_u32_avx512f);
    }

    (1, nop_filter)
}

fn bulk_u64() -> Bulk<u64> {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    if is_avx512_enabled() {
        return (8, avx512::filter_u64_avx512f);
    }

    (1, nop_filter)
}

pub fn filter_values<T: Pod>(values: &[T], mask: &Bitmap) -> Vec<T> {
    let mut out = Vec::new();
    filter_values_into(&mut out, values, mask);
    out
}

/// Appends the values of `values` selected by `mask` to `out`.
pub fn filter_values_into<T: Pod>(out: &mut Vec<T>, values: &[T], mask: &Bitmap) {
    match (size_of::<T>(), align_of::<T>()) {
        (1, 1) => filter_into(out, values, mask, bulk_u8()),
        (2, 2) => filter_into(out, values, mask, bulk_u16()),
        (4, 4) => filter_into(out, values, mask, bulk_u32()),
        (8, 8) => filter_into(out, values, mask, bulk_u64()),
        _ => filter_into::<T, T>(out, values, mask, (1, nop_filter)),
    }
}

/// `L` is the type the kernels work on and must have the same size and alignment as `T`.
fn filter_into<T: Pod, L: Pod>(
    out: &mut Vec<T>,
    values: &[T],
    mask: &Bitmap,
    (pad, bulk_filter): Bulk<L>,
) {
    assert_eq!(values.len(), mask.len());
    assert_eq!(size_of::<T>(), size_of::<L>());
    assert_eq!(align_of::<T>(), align_of::<L>());

    let mask_bits_set = mask.set_bits();
    let start = out.len();
    out.reserve(mask_bits_set + pad);

    unsafe {
        let out_ptr = out.as_mut_ptr().add(start).cast::<L>();
        let values = cast_slice::<T, L>(values);
        let (values, mask_bytes, out_ptr) = scalar_filter_offset(values, mask, out_ptr);
        let (values, mask_bytes, out_ptr) = bulk_filter(values, mask_bytes, out_ptr);
        scalar_filter(values, mask_bytes, out_ptr);
        out.set_len(start + mask_bits_set);
    }
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
