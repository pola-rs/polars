use bytemuck::{Pod, cast_slice};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::bitmap::bitmask::BitMask;
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
    if out.capacity() - start < mask_bits_set {
        out.reserve(mask_bits_set + pad);
    }

    // The kernels write up to `pad` values past the last value they keep. When
    // that does not fit, the last values are appended one by one instead.
    let spare = out.capacity() - start;
    let num_bulk = mask_bits_set.min(spare.saturating_sub(pad));
    let bulk_len = if spare >= mask_bits_set + pad {
        values.len()
    } else if num_bulk == 0 {
        0
    } else {
        BitMask::from_bitmap(mask)
            .nth_set_bit_idx_rev(mask_bits_set - num_bulk - 1, mask.len())
            .unwrap()
    };
    let bulk_mask;
    let bulk_mask = if bulk_len == values.len() {
        mask
    } else {
        bulk_mask = mask.clone().sliced(0, bulk_len);
        &bulk_mask
    };

    unsafe {
        let out_ptr = out.as_mut_ptr().add(start).cast::<L>();
        let values = cast_slice::<T, L>(&values[..bulk_len]);
        let (values, mask_bytes, out_ptr) = scalar_filter_offset(values, bulk_mask, out_ptr);
        let (values, mask_bytes, out_ptr) = bulk_filter(values, mask_bytes, out_ptr);
        scalar_filter(values, mask_bytes, out_ptr);
        out.set_len(start + num_bulk);
    }

    if bulk_len < values.len() {
        let rest_mask = mask.clone().sliced(bulk_len, values.len() - bulk_len);
        let rest = values[bulk_len..].iter().zip(rest_mask.iter());
        out.extend(rest.filter_map(|(v, keep)| keep.then_some(*v)));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(len: usize, density: f64, seed: u64) -> Bitmap {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f64 / (1u64 << 31) as f64) < density
            })
            .collect()
    }

    /// Appends several masked chunks into a buffer that has exactly the room
    /// needed, as the parquet decoder does, and checks it never grows.
    fn check_exact_capacity<T: Pod + PartialEq + std::fmt::Debug>(to_value: impl Fn(usize) -> T) {
        for len in [0, 1, 7, 8, 63, 64, 65, 200, 1000] {
            for density in [0.0, 0.05, 0.5, 0.98, 1.0] {
                for offset in [0, 3] {
                    let chunks: Vec<(Vec<T>, Bitmap)> = (0..3)
                        .map(|c| {
                            let values = (0..len).map(|i| to_value(c * len + i)).collect();
                            let mask =
                                mask(len + offset, density, (c + len) as u64).sliced(offset, len);
                            (values, mask)
                        })
                        .collect();

                    let mut expected = Vec::new();
                    for (values, mask) in &chunks {
                        expected
                            .extend(values.iter().zip(mask.iter()).filter(|x| x.1).map(|x| *x.0));
                    }

                    let mut out = Vec::with_capacity(expected.len());
                    for (values, mask) in &chunks {
                        filter_values_into(&mut out, values, mask);
                    }
                    assert_eq!(out, expected, "len={len} density={density} offset={offset}");
                    assert_eq!(
                        out.capacity(),
                        expected.len(),
                        "len={len} density={density} offset={offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_filter_values_into_exact_capacity() {
        check_exact_capacity(|i| i as u8);
        check_exact_capacity(|i| i as u16);
        check_exact_capacity(|i| i as u32);
        check_exact_capacity(|i| i as u64);
        check_exact_capacity(|i| i as u128);
    }
}
