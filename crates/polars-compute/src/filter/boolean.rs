use super::*;

fn pext32_polyfill(v: u32, m: u32) -> u32 {
    todo!()
}

fn filter_boolean_kernel(values: &Bitmap, mask: &Bitmap) -> Bitmap {
    assert_eq!(values.len(), mask.len());
    
    // Fast path: values is all-0s or all-1s.
    if let Some(num_values_bits) = values.lazy_set_bits() {
        if num_values_bits == 0 || num_values_bits == values.len() {
            return Bitmap::new_with_value(num_values_bits == values.len(), mask.set_bits());
        }
    }

    // Fast path: mask is all-0s or all-1s.
    let mask_bits_set = mask.set_bits();
    if mask_bits_set == 0 {
        return Bitmap::new();
    } else if mask_bits_set == mask.len() {
        return values.clone();
    }

    // Overallocate by 1 u64 so we can always do a full u64 write.
    let words = mask.set_bits().div_ceil(64) + 1;
    let mut out_vec: Vec<u8> = Vec::with_capacity(words * 8);
    let mut out_ptr = out_vec.as_mut_ptr();
    let mut word_offset = 0usize;
    let mut word = 0u64;

    if polars_utils::cpuid::has_fast_bmi2() {
        #[cfg(target_feature = "bmi2")]
        for (v, m) in values.fast_iter_u32().zip(mask.fast_iter_u32()) {
            // Fast-path, all-0 mask.
            // PEXT is so fast that this is the only fast-path that is worth it.
            if m == 0 {
                continue;
            }

            let mask_popcnt = m.count_ones();
            let bits = unsafe { core::arch::x86_64::_pext_u32(v, m) };

            word |= (bits as u64) << word_offset;
            word_offset += mask_popcnt as usize;
            unsafe {
                out_ptr.cast::<u64>().write_unaligned(word.to_le());
                
                let written = word_offset / 8;
                out_ptr = out_ptr.add(written);
                word_offset = word_offset % 8;
                word >>= written;
            }
        }
    } else {
        for (v, m) in values.fast_iter_u32().zip(mask.fast_iter_u32()) {
            // Fast-path, all-0 mask.
            if m == 0 {
                continue;
            }

            // Fast path, all-1 mask.
            if m == u32::MAX {
                unsafe {
                    word |= (v as u64) << word_offset;
                    out_ptr.cast::<u64>().write_unaligned(word.to_le());
                    out_ptr = out_ptr.add(4);
                    word >>= 32;
                }
                continue;
            }

            let mask_popcnt = m.count_ones();
            
            // Fast path, value is all-0s or all-1s.
            let bits = if v == 0 || v == u32::MAX {
                v & ((1 << mask_popcnt) - 1)
            } else {
                // Slow path.
                pext32_polyfill(v, m)
            };

            word |= (bits as u64) << word_offset;
            word_offset += mask_popcnt as usize;
            unsafe {
                out_ptr.cast::<u64>().write_unaligned(word.to_le());
                
                let written = word_offset / 8;
                out_ptr = out_ptr.add(written);
                word_offset = word_offset % 8;
                word >>= written;
            }
        }
    }

    unsafe {
        out_vec.set_len(words * 8);
    }
    Bitmap::from_u8_vec(out_vec, mask.set_bits())
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
