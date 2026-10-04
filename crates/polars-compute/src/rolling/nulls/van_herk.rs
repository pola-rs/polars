use polars_utils::min_max::MinMaxPolicy;

use super::super::van_herk::{recenter_trailing, rolling_minmax_centered, van_herk_kernel};
use super::*;

pub(super) fn rolling_minmax_van_herk_nulls<T, P>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
) -> ArrayRef
where
    T: NativeType + IsFloat + Bounded,
    P: MinMaxPolicy,
{
    let values = arr.values().as_slice();
    let validity = arr.validity();
    let n = values.len();

    if window_size == 0 {
        // Every window is empty, so every output is null.
        return PrimitiveArray::<T>::new_null(T::PRIMITIVE.into(), n).boxed();
    }

    let shift = if center {
        window_size.div_ceil(2) - 1
    } else {
        0
    };

    let validity_words = validity.filter(|v| v.unset_bits() > 0).map(bitmap_words);
    let mut out = match validity_words.as_deref() {
        Some(validity_words) => van_herk_kernel::<true, T, P>(values, validity_words, window_size),
        None => rolling_minmax_centered::<T, P>(values, window_size, shift),
    };
    if shift > 0
        && let Some(validity_words) = validity_words.as_deref()
    {
        recenter_trailing::<true, T, P>(&mut out, values, validity_words, window_size, shift);
    }

    // min_periods == 0 still leaves all-null windows invalid.
    let threshold = min_periods.max(1);
    let offset_fn = match center {
        true => det_offsets_center,
        false => det_offsets,
    };
    let out_validity: Bitmap = match validity {
        Some(validity) => sliding_count_validity(validity, n, shift, window_size, threshold),
        None => create_validity(threshold, n, window_size, offset_fn, None, center)
            .unwrap_or_else(|| MutableBitmap::from_len_set(n))
            .into(),
    };

    Box::new(PrimitiveArray::<T>::new(
        T::PRIMITIVE.into(),
        out.into(),
        Some(out_validity),
    ))
}

fn bitmap_words(bitmap: &Bitmap) -> Vec<u64> {
    let mut chunks = bitmap.fast_iter_u64();
    let mut words = Vec::with_capacity(bitmap.len().div_ceil(64));
    words.extend(&mut chunks);
    let (remainder, remainder_len) = chunks.remainder();
    words.extend_from_slice(&remainder[..remainder_len.div_ceil(64)]);
    words
}

fn sliding_count_validity(
    validity: &Bitmap,
    n: usize,
    shift: usize,
    w: usize,
    threshold: usize,
) -> Bitmap {
    let total = n + shift;
    // Every window covers the full input.
    if shift + 1 >= n && w >= total {
        let set = n - validity.unset_bits();
        return Bitmap::new_with_value(set >= threshold, n);
    }
    let w = w.min(total);

    // A set bit makes `w` output positions valid.
    if threshold == 1 {
        return dilate(validity, w, total).sliced(shift, n);
    }

    // Full windows are valid only when no null reaches them.
    if threshold == w {
        let nulls_reach = dilate(&!validity, w, total);
        let mut full = MutableBitmap::with_capacity(total);
        full.extend_constant((w - 1).min(total), false);
        full.extend_constant(n.saturating_sub(w - 1), true);
        full.extend_constant(total - full.len(), false);
        let full: Bitmap = full.into();
        return polars_arrow::bitmap::and_not(&full, &nulls_reach).sliced(shift, n);
    }

    // General case: two running popcounts over the virtual input.
    let mut buf: Vec<u8> = Vec::with_capacity(n.div_ceil(8));
    let (mut byte, mut filled) = (0u8, 0u8);
    let (mut hi, mut lo) = (0usize, 0usize);
    for v in 0..total {
        if v < n {
            // SAFETY: `v < n` is checked above.
            hi += unsafe { validity.get_bit_unchecked(v) } as usize;
        }
        if v >= w {
            // SAFETY: `v < n + shift` and `w > shift` here.
            lo += unsafe { validity.get_bit_unchecked(v - w) } as usize;
        }
        if v >= shift {
            byte |= ((hi - lo >= threshold) as u8) << filled;
            if filled == 7 {
                buf.push(byte);
                (byte, filled) = (0, 0);
            } else {
                filled += 1;
            }
        }
    }
    if filled > 0 {
        buf.push(byte);
    }
    MutableBitmap::from_vec(buf, n).into()
}

/// Dilate set bits to the following `w.saturating_sub(1)` positions.
///
/// `out[i] = OR_{j=max(0, i-w+1)..=i} in[j]`, i.e. every set bit is smeared to cover a
/// window of `w` positions ending at itself. The result is extended with unset bits to
/// `out_len`, which must be at least `bitmap.len()`. Computed by repeated doubling: each
/// round ORs the words with themselves shifted left by `min(covered, w - covered)`, so
/// after a round `covered` positions are covered, turning an O(w) smear into O(log w)
/// rounds.
fn dilate(bitmap: &Bitmap, w: usize, out_len: usize) -> Bitmap {
    assert!(out_len >= bitmap.len());
    let num_words = out_len.div_ceil(64);
    let mut words = bitmap_words(bitmap);
    words.resize(num_words, 0);

    // Double the covered range each round, in place: only pre-round values are read.
    let mut covered = 1usize;
    while covered < w {
        let shift = covered.min(w - covered);
        covered += shift;
        let (word_shift, bit_shift) = (shift / 64, (shift % 64) as u32);
        if word_shift == 0 {
            let inverse_shift = 64 - bit_shift;
            let mut previous_word = 0u64;
            for word in &mut words {
                let current_word = *word;
                *word |= previous_word >> inverse_shift | current_word << bit_shift;
                previous_word = current_word;
            }
        } else {
            let previous = words.clone();
            if bit_shift == 0 {
                for (dst, &src) in words.iter_mut().skip(word_shift).zip(&previous) {
                    *dst |= src;
                }
            } else {
                let inverse_shift = 64 - bit_shift;
                for (dst, src) in words
                    .iter_mut()
                    .skip(word_shift + 1)
                    .zip(previous.array_windows::<2>())
                {
                    *dst |= src[0] >> inverse_shift | src[1] << bit_shift;
                }
                if word_shift < words.len() {
                    words[word_shift] |= words[0] << bit_shift;
                }
            }
        }
    }

    // No-op on little-endian targets; the buffer is read as little-endian bytes.
    for word in &mut words {
        *word = word.to_le();
    }
    Bitmap::from_u8_vec(bytemuck::cast_slice(&words).to_vec(), out_len)
}

#[cfg(test)]
mod tests {
    use polars_arrow::bitmap::Bitmap;
    use polars_arrow::bitmap::proptest::bitmap;
    use proptest::prelude::*;

    use super::dilate;

    fn sliced_bitmap() -> impl Strategy<Value = Bitmap> {
        bitmap(1..300).prop_flat_map(|b| {
            (0..b.len(), 1..=b.len()).prop_map(move |(offset, len)| {
                let len = len.min(b.len() - offset);
                b.clone().sliced(offset, len)
            })
        })
    }

    proptest! {
        #[test]
        fn test_dilate(
            bitmap in sliced_bitmap(),
            w in 0..300usize,
            extra in 0..300usize,
        ) {
            let out_len = bitmap.len() + extra;
            let out = dilate(&bitmap, w, out_len);
            let expected: Bitmap = (0..out_len)
                .map(|i| {
                    let start = i.saturating_sub(w.saturating_sub(1));
                    (start..=i).any(|j| j < bitmap.len() && bitmap.get_bit(j))
                })
                .collect();
            prop_assert_eq!(out, expected);
        }
    }
}
