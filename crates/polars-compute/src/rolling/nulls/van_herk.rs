use polars_utils::min_max::MinMaxPolicy;

use super::super::van_herk::{recenter_trailing, rolling_minmax_pick};
use super::*;

pub(super) fn rolling_minmax_van_herk_nulls<const MIN: bool, T, P>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
) -> ArrayRef
where
    T: NativeType + PartialOrd + IsFloat + Bounded,
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

    let validity_words = validity
        .filter(|v| v.unset_bits() > 0)
        .map(collect_validity_words);
    let mut out = match validity_words.as_deref() {
        Some(validity_words) => {
            rolling_minmax_pick::<MIN, true, T, P>(values, validity_words, window_size)
        },
        None => {
            super::super::van_herk::rolling_minmax_centered::<MIN, T, P>(values, window_size, shift)
        },
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

fn collect_validity_words(validity: &Bitmap) -> Vec<u64> {
    let mut chunks = validity.fast_iter_u64();
    let mut validity_words = Vec::with_capacity(validity.len().div_ceil(64));
    validity_words.extend(&mut chunks);
    let (remainder, remainder_len) = chunks.remainder();
    validity_words.extend_from_slice(&remainder[..remainder_len.div_ceil(64)]);
    validity_words
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
        return validity.dilate(w, total).sliced(shift, n);
    }

    // Full windows are valid only when no null reaches them.
    if threshold == w {
        let nulls_reach = (!validity).dilate(w, total);
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
