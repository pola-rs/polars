use std::cell::Cell;

use polars_arrow::bitmap::BitmapBuilder;
use polars_arrow::bitmap::bitmask::BitMask;
use polars_utils::min_max::MinMaxPolicy;

use super::super::van_herk::{rolling_minmax_shifted, shift_trailing, van_herk_kernel};
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

    let masked_validity = validity.filter(|v| v.unset_bits() > 0);
    let mut out = match masked_validity {
        Some(validity) => van_herk_kernel::<true, T, P>(values, validity, window_size),
        None => rolling_minmax_shifted::<T, P>(values, window_size, shift),
    };
    if shift > 0
        && let Some(validity) = masked_validity
    {
        shift_trailing::<true, T, P>(&mut out, values, validity, window_size, shift);
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

    // General case: batch entering and leaving bits in 32-bit words.
    if n - validity.unset_bits() < threshold {
        return Bitmap::new_zeroed(n);
    }
    let mask = BitMask::from_bitmap(validity);
    let mut count = shift - validity.null_count_range(0, shift);
    let mut bytes = Vec::with_capacity(n.div_ceil(8));
    for i in (0..n).step_by(32) {
        let v = shift + i;
        let mut incoming = mask.get_u32(v);
        // Zero-pad outgoing bits before the window reaches the input.
        let mut outgoing = mask
            .get_u32(v.saturating_sub(w))
            .checked_shl(w.saturating_sub(v).min(32) as u32)
            .unwrap_or(0);
        let len = (n - i).min(32);
        let mut word = 0u32;
        for bit in 0..len {
            count += (incoming & 1) as usize;
            incoming >>= 1;
            count -= (outgoing & 1) as usize;
            outgoing >>= 1;
            word |= ((count >= threshold) as u32) << bit;
        }
        bytes.extend_from_slice(&word.to_le_bytes()[..len.div_ceil(8)]);
    }
    Bitmap::from_u8_vec(bytes, n)
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
    let padded_len = out_len.div_ceil(64) * 64;
    let mut out = BitmapBuilder::with_capacity(padded_len);
    out.extend_from_bitmap(bitmap);
    out.extend_constant(padded_len - bitmap.len(), false);
    let mut out = out.into_mut();
    let words = out.as_mut_slice().as_chunks_mut::<8>().0;

    // Carry original words forward for sub-word shifts; walk backward for larger shifts.
    // Earlier doubling already covers a mixed shift's first destination word.
    let mut covered = 1usize;
    while covered < w {
        let shift = covered.min(w - covered);
        covered += shift;
        let (word_shift, bit_shift) = (shift / 64, shift % 64);
        let inverse_shift = 64 - bit_shift;
        if word_shift == 0 {
            let mut previous_word = 0u64;
            for word in words.iter_mut() {
                let current_word = u64::from_le_bytes(*word);
                *word = (current_word | previous_word >> inverse_shift | current_word << bit_shift)
                    .to_le_bytes();
                previous_word = current_word;
            }
        } else if bit_shift == 0 {
            // Cells allow overlapping source and destination views without copying.
            let cells = Cell::from_mut(words).as_slice_of_cells();
            for (dst, src) in cells.iter().skip(word_shift).zip(cells).rev() {
                dst.set(std::array::from_fn(|i| dst.get()[i] | src.get()[i]));
            }
        } else {
            for i in (word_shift + 1..words.len()).rev() {
                let smeared = u64::from_le_bytes(words[i - word_shift]) << bit_shift
                    | u64::from_le_bytes(words[i - word_shift - 1]) >> inverse_shift;
                words[i] = (u64::from_le_bytes(words[i]) | smeared).to_le_bytes();
            }
        }
    }
    out.resize(out_len, false);
    out.into()
}
