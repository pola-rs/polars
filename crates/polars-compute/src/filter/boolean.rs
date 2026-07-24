use arrow::bitmap::bitmask::BitMask;
use arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_utils::clmul::prefix_xorsum;

const U56_MAX: u64 = (1 << 56) - 1;

fn pext64_polyfill(mut v: u64, mut m: u64, m_popcnt: u32) -> u64 {
    // Fast path: popcount is low.
    if m_popcnt <= 4 {
        // Not a "while m != 0" but a for loop instead so the compiler fully
        // unrolls the loop, this makes bit << i much faster.
        let mut out = 0;
        for i in 0..4 {
            if m == 0 {
                break;
            };

            let bit = (v >> m.trailing_zeros()) & 1;
            out |= bit << i;
            m &= m.wrapping_sub(1); // Clear least significant bit.
        }
        return out;
    }

    // Fast path: all the masked bits in v are 0 or 1.
    // Despite this fast path being simpler than the above popcount-based one,
    // we do it afterwards because if m has a low popcount these branches become
    // very unpredictable.
    v &= m;
    if v == 0 {
        return 0;
    } else if v == m {
        return (1 << m_popcnt) - 1;
    }

    // This algorithm is too involved to explain here, see https://github.com/zwegner/zp7.
    // That is an optimized version of Hacker's Delight Chapter 7-4, parallel suffix method for compress().
    let mut invm = !m;

    for i in 0..6 {
        let shift = 1 << i;
        let prefix_count_bit = if i < 5 {
            prefix_xorsum(invm)
        } else {
            invm.wrapping_neg() << 1
        };
        let keep_in_place = v & !prefix_count_bit;
        let shift_down = v & prefix_count_bit;
        v = keep_in_place | (shift_down >> shift);
        invm &= prefix_count_bit;
    }
    v
}

fn pdep64_polyfill(mut v: u64, mut m: u64, m_popcnt: u32) -> u64 {
    // Empirically determined crossover for fast path.
    if m_popcnt <= 8 {
        let mut out = 0;
        while m != 0 {
            let lowest_mask_bit = m & m.wrapping_neg();
            out |= lowest_mask_bit & (v & 1).wrapping_neg();
            v >>= 1;
            m &= m - 1;
        }
        return out;
    }
    // As for pext, see https://github.com/zwegner/zp7. This is Hacker's
    // Delight 7-5, parallel suffix method for expand.
    v &= u64::MAX >> (64 - m_popcnt);
    let mut invm = !m;
    let mut prefix_count_bits = [0; 6];
    for (i, prefix_count_bit) in prefix_count_bits.iter_mut().enumerate() {
        *prefix_count_bit = if i < 5 {
            prefix_xorsum(invm)
        } else {
            invm.wrapping_neg() << 1
        };
        invm &= *prefix_count_bit;
    }

    for i in (0..6).rev() {
        let shift = 1 << i;
        let bit = prefix_count_bits[i] >> shift;
        v = (v & !bit) + ((v & bit) << shift);
    }
    v
}

pub fn filter_boolean_kernel(values: &Bitmap, mask: &Bitmap) -> Bitmap {
    assert_eq!(values.len(), mask.len());
    let mask_bits_set = mask.set_bits();

    // Fast path: values is all-0s or all-1s.
    if let Some(num_values_bits) = values.lazy_set_bits() {
        if num_values_bits == 0 || num_values_bits == values.len() {
            return Bitmap::new_with_value(num_values_bits == values.len(), mask_bits_set);
        }
    }

    // Fast path: mask is all-0s or all-1s.
    if mask_bits_set == 0 {
        return Bitmap::new();
    } else if mask_bits_set == mask.len() {
        return values.clone();
    }

    // Overallocate by 1 u64 so we can always do a full u64 write.
    let num_words = mask_bits_set.div_ceil(64);
    let num_bytes = 8 * (num_words + 1);
    let mut out_vec: Vec<u8> = Vec::with_capacity(num_bytes);

    unsafe {
        if mask_bits_set <= mask.len() / (64 * 4) {
            // Less than one in 1 in 4 words has a bit set on average, use sparse kernel.
            filter_boolean_kernel_sparse(values, mask, out_vec.as_mut_ptr());
        } else if polars_utils::cpuid::has_fast_bmi2() {
            #[cfg(target_arch = "x86_64")]
            filter_boolean_kernel_pext::<true, _>(values, mask, out_vec.as_mut_ptr(), |v, m, _| {
                // SAFETY: has_fast_bmi2 ensures this is a legal instruction.
                core::arch::x86_64::_pext_u64(v, m)
            });
        } else {
            filter_boolean_kernel_pext::<false, _>(
                values,
                mask,
                out_vec.as_mut_ptr(),
                pext64_polyfill,
            )
        }

        // SAFETY: the above filters must have initialized these bytes.
        out_vec.set_len(mask_bits_set.div_ceil(8));
    }

    Bitmap::from_u8_vec(out_vec, mask_bits_set)
}

fn expand_boolean_kernel_impl<F: Fn(u64, u64, u32) -> u64>(
    values: &Bitmap,
    mask: &Bitmap,
    pdep: F,
) -> Bitmap {
    let values = BitMask::from_bitmap(values);
    let mut values_offset = 0;
    let mut builder = BitmapBuilder::with_capacity(mask.len());

    let mut mask_chunks = mask.chunks::<u64>();
    for mask_word in &mut mask_chunks {
        let mask_popcnt = mask_word.count_ones();
        let values_word = values.get_u64(values_offset);
        values_offset += mask_popcnt as usize;

        // SAFETY: the builder has capacity for `mask.len()` bits and every full mask chunk adds
        // exactly 64 bits.
        unsafe {
            builder.push_word_with_len_unchecked(pdep(values_word, mask_word, mask_popcnt), 64)
        };
    }

    let remainder_len = mask_chunks.remainder_len();
    if remainder_len > 0 {
        let mask_word = mask_chunks.remainder();
        let mask_popcnt = mask_word.count_ones();
        let values_word = values.get_u64(values_offset);
        values_offset += mask_popcnt as usize;

        // SAFETY: `remainder_len` is the exact unused capacity and is at most 63 bits.
        unsafe {
            builder.push_word_with_len_unchecked(
                pdep(values_word, mask_word, mask_popcnt),
                remainder_len,
            )
        };
    }

    debug_assert_eq!(values_offset, values.len());
    builder.freeze()
}

/// Expands the bits in `values` into the set positions of `mask`.
///
/// The returned bitmap has the length of `mask`; positions where `mask` is unset are also unset in
/// the result. The number of bits in `values` must equal the number of set bits in `mask`.
pub fn expand_boolean_kernel(values: &Bitmap, mask: &Bitmap) -> Bitmap {
    let mask_bits_set = mask.set_bits();
    assert_eq!(values.len(), mask_bits_set);

    if mask_bits_set == 0 {
        return Bitmap::new_zeroed(mask.len());
    } else if mask_bits_set == mask.len() {
        return values.clone();
    }

    if let Some(num_value_bits) = values.lazy_set_bits() {
        if num_value_bits == 0 {
            return Bitmap::new_zeroed(mask.len());
        } else if num_value_bits == values.len() {
            return mask.clone();
        }
    }

    if polars_utils::cpuid::has_fast_bmi2() {
        #[cfg(target_arch = "x86_64")]
        return expand_boolean_kernel_impl(values, mask, |v, m, _| unsafe {
            // SAFETY: has_fast_bmi2 ensures this is a legal instruction.
            core::arch::x86_64::_pdep_u64(v, m)
        });
    }

    expand_boolean_kernel_impl(values, mask, pdep64_polyfill)
}

/// # Safety
/// out_ptr must point to a buffer of length >= 8 + 8 * ceil(mask.set_bits() / 64).
/// This function will initialize at least the first ceil(mask.set_bits() / 8) bytes.
unsafe fn filter_boolean_kernel_sparse(values: &Bitmap, mask: &Bitmap, mut out_ptr: *mut u8) {
    assert_eq!(values.len(), mask.len());

    let mut value_idx = 0;
    let mut bits_in_word = 0usize;
    let mut word = 0u64;

    macro_rules! loop_body {
        ($m: expr) => {{
            let mut m = $m;
            while m > 0 {
                let idx_in_m = m.trailing_zeros() as usize;
                let bit = unsafe { values.get_bit_unchecked(value_idx + idx_in_m) };
                word |= (bit as u64) << bits_in_word;
                bits_in_word += 1;

                if bits_in_word == 64 {
                    unsafe {
                        out_ptr.cast::<u64>().write_unaligned(word.to_le());
                        out_ptr = out_ptr.add(8);
                        bits_in_word = 0;
                        word = 0;
                    }
                }

                m &= m.wrapping_sub(1); // Clear least significant bit.
            }
        }};
    }

    let mask_aligned = mask.aligned::<u64>();
    if mask_aligned.prefix_bitlen() > 0 {
        loop_body!(mask_aligned.prefix());
        value_idx += mask_aligned.prefix_bitlen();
    }

    for m in mask_aligned.bulk_iter() {
        loop_body!(m);
        value_idx += 64;
    }

    if mask_aligned.suffix_bitlen() > 0 {
        loop_body!(mask_aligned.suffix());
    }

    if bits_in_word > 0 {
        unsafe {
            out_ptr.cast::<u64>().write_unaligned(word.to_le());
        }
    }
}

/// # Safety
/// See filter_boolean_kernel_sparse.
unsafe fn filter_boolean_kernel_pext<const HAS_NATIVE_PEXT: bool, F: Fn(u64, u64, u32) -> u64>(
    values: &Bitmap,
    mask: &Bitmap,
    mut out_ptr: *mut u8,
    pext: F,
) {
    assert_eq!(values.len(), mask.len());

    let mut bits_in_word = 0usize;
    let mut word = 0u64;

    macro_rules! loop_body {
        ($v: expr, $m: expr) => {{
            let (v, m) = ($v, $m);

            // Fast-path, all-0 mask.
            if m == 0 {
                continue;
            }

            // Fast path, all-1 mask.
            // This is only worth it if we don't have a native pext.
            if !HAS_NATIVE_PEXT && m == U56_MAX {
                word |= v << bits_in_word;
                unsafe {
                    out_ptr.cast::<u64>().write_unaligned(word.to_le());
                    out_ptr = out_ptr.add(7);
                }
                word >>= 56;
                continue;
            }

            let mask_popcnt = m.count_ones();
            let bits = pext(v, m, mask_popcnt);

            // Because we keep bits_in_word < 8 and we iterate over u56s,
            // this never loses output bits.
            word |= bits << bits_in_word;
            bits_in_word += mask_popcnt as usize;
            unsafe {
                out_ptr.cast::<u64>().write_unaligned(word.to_le());

                let full_bytes_written = bits_in_word / 8;
                out_ptr = out_ptr.add(full_bytes_written);
                word >>= full_bytes_written * 8;
                bits_in_word %= 8;
            }
        }};
    }

    let mut v_iter = values.fast_iter_u56();
    let mut m_iter = mask.fast_iter_u56();
    for v in &mut v_iter {
        // SAFETY: we checked values and mask have same length.
        let m = unsafe { m_iter.next().unwrap_unchecked() };
        loop_body!(v, m);
    }
    let mut v_rem = v_iter.remainder().0;
    let mut m_rem = m_iter.remainder().0;
    while m_rem != 0 {
        let v = v_rem & U56_MAX;
        let m = m_rem & U56_MAX;
        v_rem >>= 56;
        m_rem >>= 56;
        loop_body!(v, m); // Careful, contains 'continue', increment loop variables first.
    }
}

pub fn filter_bitmap_and_validity(
    values: &Bitmap,
    validity: Option<&Bitmap>,
    mask: &Bitmap,
) -> (Bitmap, Option<Bitmap>) {
    let filtered_values = filter_boolean_kernel(values, mask);
    if let Some(validity) = validity {
        // TODO: we could theoretically be faster by computing these two filters
        // at once. Unsure if worth duplicating all the code above.
        let filtered_validity = filter_boolean_kernel(validity, mask);
        (filtered_values, Some(filtered_validity))
    } else {
        (filtered_values, None)
    }
}

#[cfg(test)]
mod test {
    use rand::prelude::*;

    use super::*;

    fn naive_pext64(word: u64, mask: u64) -> u64 {
        let mut out = 0;
        let mut out_idx = 0;

        for i in 0..64 {
            let ith_mask_bit = (mask >> i) & 1;
            let ith_word_bit = (word >> i) & 1;
            if ith_mask_bit == 1 {
                out |= ith_word_bit << out_idx;
                out_idx += 1;
            }
        }

        out
    }

    fn naive_pdep64(word: u64, mask: u64) -> u64 {
        let mut out = 0;
        let mut word_idx = 0;

        for i in 0..64 {
            if (mask >> i) & 1 == 1 {
                out |= ((word >> word_idx) & 1) << i;
                word_idx += 1;
            }
        }

        out
    }

    fn naive_expand(values: &Bitmap, mask: &Bitmap) -> Bitmap {
        assert_eq!(values.len(), mask.set_bits());
        let mut values = values.iter();
        mask.iter()
            .map(|selected| selected && values.next().unwrap())
            .collect()
    }

    #[test]
    fn test_pext64() {
        // Verify polyfill against naive implementation.
        let mut rng = StdRng::seed_from_u64(0xdeadbeef);
        for _ in 0..100 {
            let x = rng.random();
            let y = rng.random();
            assert_eq!(naive_pext64(x, y), pext64_polyfill(x, y, y.count_ones()));

            // Test all-zeros and all-ones.
            assert_eq!(naive_pext64(0, y), pext64_polyfill(0, y, y.count_ones()));
            assert_eq!(
                naive_pext64(u64::MAX, y),
                pext64_polyfill(u64::MAX, y, y.count_ones())
            );
            assert_eq!(naive_pext64(x, 0), pext64_polyfill(x, 0, 0));
            assert_eq!(naive_pext64(x, u64::MAX), pext64_polyfill(x, u64::MAX, 64));

            // Test low popcount mask.
            let popcnt = rng.random_range(0..=8);
            // Not perfect (can generate same bit twice) but it'll do.
            let mask = (0..popcnt).map(|_| 1 << rng.random_range(0..64)).sum();
            assert_eq!(
                naive_pext64(x, mask),
                pext64_polyfill(x, mask, mask.count_ones())
            );
        }
    }

    #[test]
    fn test_pdep64() {
        let mut rng = StdRng::seed_from_u64(0xdeadbeef);
        for _ in 0..100 {
            let x = rng.random();
            let y: u64 = rng.random();
            assert_eq!(naive_pdep64(x, y), pdep64_polyfill(x, y, y.count_ones()));

            assert_eq!(naive_pdep64(0, y), pdep64_polyfill(0, y, y.count_ones()));
            assert_eq!(
                naive_pdep64(u64::MAX, y),
                pdep64_polyfill(u64::MAX, y, y.count_ones())
            );
            assert_eq!(naive_pdep64(x, 0), pdep64_polyfill(x, 0, 0));
            assert_eq!(naive_pdep64(x, u64::MAX), pdep64_polyfill(x, u64::MAX, 64));
        }
    }

    #[test]
    fn test_expand_boolean_kernel() {
        let mask: Bitmap = [false, true, true, false, true, false]
            .into_iter()
            .collect();
        let values: Bitmap = [true, false, true].into_iter().collect();
        let expected: Bitmap = [false, true, false, false, true, false]
            .into_iter()
            .collect();
        assert_eq!(expand_boolean_kernel(&values, &mask), expected);

        let mut rng = StdRng::seed_from_u64(0xc0ffee);
        for len in [0, 1, 7, 8, 55, 56, 57, 63, 64, 65, 127, 128, 129, 257] {
            for mask_offset in 0..8 {
                let mask_storage: Bitmap =
                    (0..len + mask_offset + 8).map(|_| rng.random()).collect();
                let mask = mask_storage.sliced(mask_offset, len);
                let num_values = mask.set_bits();

                for values_offset in 0..8 {
                    let values_storage: Bitmap = (0..num_values + values_offset + 8)
                        .map(|_| rng.random())
                        .collect();
                    let values = values_storage.sliced(values_offset, num_values);
                    let expected = naive_expand(&values, &mask);

                    assert_eq!(expand_boolean_kernel(&values, &mask), expected);
                    assert_eq!(
                        expand_boolean_kernel_impl(&values, &mask, pdep64_polyfill),
                        expected,
                    );
                    assert_eq!(filter_boolean_kernel(&expected, &mask), values);
                }

                let original_storage: Bitmap =
                    (0..len + mask_offset + 8).map(|_| rng.random()).collect();
                let original = original_storage.sliced(mask_offset, len);
                let compressed = filter_boolean_kernel(&original, &mask);
                let expanded = expand_boolean_kernel(&compressed, &mask);
                assert_eq!(expanded, &original & &mask);
            }
        }
    }
}
