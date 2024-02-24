use super::*;

use polars_utils::clmul::{portable_prefix_xorsum, fast_prefix_xorsum};

const U56_MAX: u64 = (1 << 56) - 1;

// TODO fast_iter remainders...


fn isolate_lsb_block(mut v: u64) -> u64 {
    let lsb = v & v.wrapping_neg();
    v & !v.wrapping_add(lsb)
}

fn pext64_polyfill<F: Fn(u64) -> u64>(mut v: u64, mut m: u64, prefix_xorsum: F) -> u64 {
    // This algorithm is too involved to explain here, see https://github.com/zwegner/zp7.
    let mut invm = !m;

    v &= m;
    for i in 0..6 {
        let shift = 1 << i;
        let prefix_count_bit = if i < 5 { prefix_xorsum(invm) } else { invm.wrapping_neg() << 1 };
        let keep_in_place = v & !prefix_count_bit;
        let shift_down = v & prefix_count_bit;
        v = keep_in_place | (shift_down >> shift);
        invm &= prefix_count_bit;
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
    let num_bytes = 8 + 8 * mask_bits_set.div_ceil(64);
    let mut out_vec: Vec<u8> = Vec::with_capacity(num_bytes);

    unsafe {
        // Make sure the tail is always initialized.
        let guaranteed_initialized = mask_bits_set.div_ceil(8);
        let num_tail_bytes = num_bytes - guaranteed_initialized;
        out_vec
            .as_mut_ptr()
            .add(num_bytes - num_tail_bytes)
            .write_bytes(0, num_tail_bytes);

        if mask_bits_set <= mask.len() / 32 {
            // Fast path: mask is very sparse, 1 in 32 bits or fewer set.
            filter_boolean_kernel_sparse(values, mask, out_vec.as_mut_ptr());
        } else if polars_utils::cpuid::has_fast_bmi2() {
            #[cfg(target_arch = "x86_64")]
            filter_boolean_kernel_pext(values, mask, out_vec.as_mut_ptr(), |v, m, _| {
                // SAFETY: has_fast_bmi2 ensures this is a legal instruction.
                unsafe { core::arch::x86_64::_pext_u64(v, m) }
            });
        } else if polars_utils::cpuid::has_fast_clmul() {
            filter_boolean_kernel_pext(values, mask, out_vec.as_mut_ptr(), |v, m, m_popcnt| {
                if v == 0 || v == U56_MAX {
                    // Fast path, value is all-0s or all-1s.
                    v & ((1 << m_popcnt) - 1)
                } else {
                    // Slow path.
                    pext64_polyfill(v, m, |x| fast_prefix_xorsum(x))
                }
            });
        } else {
            filter_boolean_kernel_pext(values, mask, out_vec.as_mut_ptr(), |v, m, m_popcnt| {
                if v == 0 || v == U56_MAX {
                    // Fast path, value is all-0s or all-1s.
                    v & ((1 << m_popcnt) - 1)
                } else {
                    // Slow path.
                    pext64_polyfill(v, m, portable_prefix_xorsum)
                }
            });
        }

        out_vec.set_len(num_bytes);
    }

    Bitmap::from_u8_vec(out_vec, mask_bits_set)
}

/// # Safety
/// out_ptr must point to a buffer of length >= 8 + 8 * ceil(mask.set_bits() / 64).
/// This function will initialize at least the first ceil(mask.set_bits() / 8) bytes.
unsafe fn filter_boolean_kernel_sparse(values: &Bitmap, mask: &Bitmap, mut out_ptr: *mut u8) {
    let mut word_offset = 0usize;
    let mut word = 0u64;
    for idx in mask.true_idx_iter() {
        word |= (values.get_bit(idx) as u64) << word_offset;
        word_offset += 1;

        if word_offset == 64 {
            unsafe {
                out_ptr.cast::<u64>().write_unaligned(word.to_le());
                out_ptr = out_ptr.add(8);
                word_offset = 0;
                word = 0;
            }
        }
    }

    if word_offset > 0 {
        unsafe {
            out_ptr.cast::<u64>().write_unaligned(word.to_le());
        }
    }
}


unsafe fn filter_boolean_kernel_clmul_pext<F: Fn(u64, u64, u32) -> u64>(values: &Bitmap, mask: &Bitmap, mut out_ptr: *mut u8, pext: F) {
    filter_boolean_kernel_pext(values, mask, out_vec.as_mut_ptr(), |v, m, m_popcnt| {
        if v == 0 || v == U56_MAX {
            // Fast path, value is all-0s or all-1s.
            v & ((1 << m_popcnt) - 1)
        } else {
            // Slow path.
            pext64_polyfill(v, m, |x| fast_prefix_xorsum(x))
        }
    });
}

/// # Safety
/// See filter_boolean_kernel_sparse.
unsafe fn filter_boolean_kernel_pext<F: Fn(u64, u64, u32) -> u64>(values: &Bitmap, mask: &Bitmap, mut out_ptr: *mut u8, pext: F) {
    let mut word_offset = 0usize;
    let mut word = 0u64;
    for (v, m) in values.fast_iter_u56().zip(mask.fast_iter_u56()) {
        // Fast-path, all-0 mask.
        if m == 0 {
            continue;
        }

        // Fast path, all-1 mask.
        if m == U56_MAX {
            word |= v << word_offset;
            unsafe {
                out_ptr.cast::<u64>().write_unaligned(word.to_le());
                out_ptr = out_ptr.add(7);
            }
            word >>= 56;
            continue;
        }

        let mask_popcnt = m.count_ones();
        let bits = pext(v, m, mask_popcnt);

        // Because we keep word_offset < 8 and we iterate over u56s,
        // this never loses output bits.
        word |= (bits as u64) << word_offset;
        word_offset += mask_popcnt as usize;
        unsafe {
            out_ptr.cast::<u64>().write_unaligned(word.to_le());

            let written = word_offset / 8;
            out_ptr = out_ptr.add(written);
            word >>= written * 8;
            word_offset = word_offset % 8;
        }
    }
}

/*

pub fn carryless_mullo64(x: u64, y: u64) -> u64 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        vmull_p64(x, y) as u64
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        _mm_cvtsi128_si64(_mm_clmulepi64_si128(
            _mm_cvtsi64_si128(x as i64),
            _mm_cvtsi64_si128(y as i64),
            0
        )) as u64
    }
}

pub fn pext(v: u64, m: u64) -> u64 {
    let mut mm = !m;
    let prefix = u64::MAX - 1;

    let mut a = v & m;
    for i in 0..5 {
        let bit = carryless_mullo64(mm, prefix);
        a = (!bit & a) | ((bit & a) >> (1 << i));
        mm &= bit;
    }
    let bit = u64::wrapping_sub(0, mm) * 2;
    a = (!bit & a) | ((bit & a) >> (1 << 5));

    a
}


pub fn pext_loop(mut v: u64, mut m: u64, lut: &[u8]) -> u64 {
    let mut out = 0u64;
    let mut popcnt = v.count_ones();
    let mut i = 0;
    while m > 0 {
        let lsb_m = m & u64::wrapping_sub(0, m);
        out = out.rotate_right(1) | (v & lsb_m > 0) as u64;
        m ^= lsb_m;
        i += 1;
    }
    out.rotate_left(popcnt)
}


*/

pub(super) fn filter_bitmap_and_validity(
    values: &Bitmap,
    validity: Option<&Bitmap>,
    mask: &Bitmap,
) -> (MutableBitmap, Option<MutableBitmap>) {
    if let Some(validity) = validity {
        let (values, validity) = null_filter(values, validity, mask);
        (values, Some(validity))
    } else {
        (nonnull_filter(values, mask), None)
    }
}

/// # Safety
/// This assumes that the `mask_chunks` contains a number of set/true items equal
/// to `filter_count`
unsafe fn nonnull_filter_impl<I>(
    values: &Bitmap,
    mut mask_chunks: I,
    filter_count: usize,
) -> MutableBitmap
where
    I: BitChunkIterExact<u64>,
{
    // TODO! we might use ChunksExact here if offset = 0.
    let mut chunks = values.chunks::<u64>();
    let mut new = MutableBitmap::with_capacity(filter_count);

    chunks
        .by_ref()
        .zip(mask_chunks.by_ref())
        .for_each(|(chunk, mask_chunk)| {
            let ones = mask_chunk.count_ones();
            let leading_ones = get_leading_ones(mask_chunk);

            if ones == leading_ones {
                let size = leading_ones as usize;
                unsafe { new.extend_from_slice_unchecked(chunk.to_ne_bytes().as_ref(), 0, size) };
                return;
            }

            let ones_iter = BitChunkOnes::from_known_count(mask_chunk, ones as usize);
            for pos in ones_iter {
                new.push_unchecked(chunk & (1 << pos) > 0);
            }
        });

    chunks
        .remainder_iter()
        .zip(mask_chunks.remainder_iter())
        .for_each(|(value, is_selected)| {
            if is_selected {
                unsafe {
                    new.push_unchecked(value);
                };
            }
        });

    new
}

/// # Safety
/// This assumes that the `mask_chunks` contains a number of set/true items equal
/// to `filter_count`
unsafe fn null_filter_impl<I>(
    values: &Bitmap,
    validity: &Bitmap,
    mut mask_chunks: I,
    filter_count: usize,
) -> (MutableBitmap, MutableBitmap)
where
    I: BitChunkIterExact<u64>,
{
    let mut chunks = values.chunks::<u64>();
    let mut validity_chunks = validity.chunks::<u64>();

    let mut new = MutableBitmap::with_capacity(filter_count);
    let mut new_validity = MutableBitmap::with_capacity(filter_count);

    chunks
        .by_ref()
        .zip(validity_chunks.by_ref())
        .zip(mask_chunks.by_ref())
        .for_each(|((chunk, validity_chunk), mask_chunk)| {
            let ones = mask_chunk.count_ones();
            let leading_ones = get_leading_ones(mask_chunk);

            if ones == leading_ones {
                let size = leading_ones as usize;

                unsafe {
                    new.extend_from_slice_unchecked(chunk.to_ne_bytes().as_ref(), 0, size);

                    // SAFETY: invariant offset + length <= slice.len()
                    new_validity.extend_from_slice_unchecked(
                        validity_chunk.to_ne_bytes().as_ref(),
                        0,
                        size,
                    );
                }
                return;
            }

            // this triggers a bitcount
            let ones_iter = BitChunkOnes::from_known_count(mask_chunk, ones as usize);
            for pos in ones_iter {
                new.push_unchecked(chunk & (1 << pos) > 0);
                new_validity.push_unchecked(validity_chunk & (1 << pos) > 0);
            }
        });

    chunks
        .remainder_iter()
        .zip(validity_chunks.remainder_iter())
        .zip(mask_chunks.remainder_iter())
        .for_each(|((value, is_valid), is_selected)| {
            if is_selected {
                unsafe {
                    new.push_unchecked(value);
                    new_validity.push_unchecked(is_valid);
                };
            }
        });

    (new, new_validity)
}

fn null_filter(
    values: &Bitmap,
    validity: &Bitmap,
    mask: &Bitmap,
) -> (MutableBitmap, MutableBitmap) {
    assert_eq!(values.len(), mask.len());
    let filter_count = mask.len() - mask.unset_bits();

    let (slice, offset, length) = mask.as_slice();
    if offset == 0 {
        let mask_chunks = BitChunksExact::<u64>::new(slice, length);
        unsafe { null_filter_impl(values, validity, mask_chunks, filter_count) }
    } else {
        let mask_chunks = mask.chunks::<u64>();
        unsafe { null_filter_impl(values, validity, mask_chunks, filter_count) }
    }
}

fn nonnull_filter(values: &Bitmap, mask: &Bitmap) -> MutableBitmap {
    assert_eq!(values.len(), mask.len());
    let filter_count = mask.len() - mask.unset_bits();

    let (slice, offset, length) = mask.as_slice();
    if offset == 0 {
        let mask_chunks = BitChunksExact::<u64>::new(slice, length);
        unsafe { nonnull_filter_impl(values, mask_chunks, filter_count) }
    } else {
        let mask_chunks = mask.chunks::<u64>();
        unsafe { nonnull_filter_impl(values, mask_chunks, filter_count) }
    }
}



#[cfg(test)]
mod test {
    use super::*;
    use rand::prelude::*;

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
    
    #[test]
    fn test_pext64() {
        // Verify polyfill against naive implementation.
        let mut rng = StdRng::seed_from_u64(0xdeadbeef);
        for _ in 0..100 {
            let x = rng.gen();
            let y = rng.gen();
            assert_eq!(naive_pext64(x, y), pext64_polyfill(x, y, portable_prefix_xorsum));
        }
    }
}