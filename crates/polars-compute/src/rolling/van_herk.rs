use polars_utils::min_max::MinMaxPolicy;

use super::*;
use crate::nan::first_nan_idx;

pub(super) fn rolling_minmax_centered<const MIN: bool, T, P>(
    values: &[T],
    w: usize,
    shift: usize,
) -> Vec<T>
where
    T: NativeType + PartialOrd + IsFloat + Bounded,
    P: MinMaxPolicy,
{
    let mut out = if T::is_float() && first_nan_idx(values).is_some() {
        // NaNs poison windows asymmetrically: use the stable combine so ties keep
        // the earliest NaN payload or signed zero.
        van_herk_kernel::<false, T, P, _>(values, &[], w, combine::<T, P>)
    } else {
        van_herk_kernel::<false, T, P, _>(values, &[], w, combine_plain::<MIN, T>)
    };
    if shift > 0 {
        recenter_trailing::<false, T, P>(&mut out, values, &[], w, shift);
    }
    out
}

/// Fold identity, including infinities for floating-point inputs.
#[inline]
pub(super) fn identity<T: NativeType + IsFloat + Bounded, P: MinMaxPolicy>() -> T {
    let (lo, hi) = if T::is_float() {
        (T::neg_inf_value(), T::pos_inf_value())
    } else {
        (T::min_value(), T::max_value())
    };
    if P::is_better(&lo, &hi) { hi } else { lo }
}

/// Stable combine: ties keep the earlier NaN payload or signed zero.
#[inline(always)]
pub(super) fn combine<T: NativeType, P: MinMaxPolicy>(earlier: T, later: T) -> T {
    if P::is_better(&later, &earlier) {
        later
    } else {
        earlier
    }
}

#[inline(always)]
pub(super) fn combine_plain<const MIN: bool, T: NativeType + PartialOrd + IsFloat>(
    earlier: T,
    later: T,
) -> T {
    let better = if MIN {
        later < earlier
    } else {
        later > earlier
    };
    if better { later } else { earlier }
}

/// Recenter trailing output with a backward suffix scan.
/// `MASKED` avoids the measurable hot-loop cost of a generic read closure.
pub(super) fn recenter_trailing<const MASKED: bool, T, P>(
    out: &mut [T],
    values: &[T],
    validity_words: &[u64],
    w: usize,
    shift: usize,
) where
    T: NativeType + PartialOrd + IsFloat + Bounded,
    P: MinMaxPolicy,
{
    let n = values.len();
    if shift < n {
        out.copy_within(shift.., 0);
    }
    let id = identity::<T, P>();
    assert!(!MASKED || validity_words.len() >= n.div_ceil(64));
    let get = |i: usize| {
        if MASKED {
            // SAFETY: the assertion above and `i < n` cover this word.
            let word = unsafe { *validity_words.get_unchecked(i / 64) };
            if word >> (i % 64) & 1 == 0 {
                return id;
            }
        }
        // SAFETY: all callers keep `i < n`.
        unsafe { *values.get_unchecked(i) }
    };
    let mut acc = id;
    // This bound also keeps `p + w - 1` from overflowing below.
    if w < n + shift {
        let hi = n + shift - w + 1;
        for i in (hi..n).rev() {
            acc = combine::<T, P>(get(i), acc);
        }
        for p in (n.saturating_sub(w - 1)..hi).rev() {
            acc = combine::<T, P>(get(p), acc);
            out[p + w - 1 - shift] = acc;
        }
    } else {
        for i in (0..n).rev() {
            acc = combine::<T, P>(get(i), acc);
        }
    }
    for t in n.max(shift)..(n + shift).min(w) {
        out[t - shift] = acc;
    }
}

/// Shared block scan; `MASKED` avoids the measurable hot-loop cost of a generic read
/// closure and is eliminated from the direct-read specialization.
pub(super) fn van_herk_kernel<const MASKED: bool, T, P, F>(
    values: &[T],
    validity_words: &[u64],
    w: usize,
    combine_el: F,
) -> Vec<T>
where
    T: NativeType + IsFloat + Bounded,
    P: MinMaxPolicy,
    F: Fn(T, T) -> T,
{
    let id = identity::<T, P>();
    let n = values.len();
    assert!(!MASKED || validity_words.len() >= n.div_ceil(64));
    let get = |i: usize| {
        if MASKED {
            // SAFETY: the assertion above and `i < n` cover this word.
            let word = unsafe { *validity_words.get_unchecked(i / 64) };
            if word >> (i % 64) & 1 == 0 {
                return id;
            }
        }
        // SAFETY: all callers keep `i < n`.
        unsafe { *values.get_unchecked(i) }
    };
    let mut ret = Vec::with_capacity(n);
    let out = &mut ret.spare_capacity_mut()[..n];
    if w >= n {
        let mut acc = id;
        for (i, o) in out.iter_mut().enumerate() {
            acc = combine_el(acc, get(i));
            o.write(acc);
        }
    } else {
        let nb = n / w;
        let r = n % w;
        let mut b_old = vec![id; w];
        let mut b_new = vec![id; w];

        for k in 0..nb {
            let base = k * w;
            let out_block = unsafe { out.get_unchecked_mut(base..base + w) };
            // Keep the independent prefix and suffix chains in flight together.
            let mut prefix = id;
            let mut suffix = id;
            for j in 0..w - 1 {
                let vf = get(base + j);
                let vb = get(base + w - 1 - j);
                prefix = combine_el(prefix, vf);
                // Right-to-left keeps the leftmost tie or NaN.
                suffix = combine_el(vb, suffix);
                out_block[j].write(combine_el(b_old[j + 1], prefix));
                b_new[w - 1 - j] = suffix;
            }
            prefix = combine_el(prefix, get(base + w - 1));
            suffix = combine_el(get(base), suffix);
            out_block[w - 1].write(prefix);
            b_new[0] = suffix;
            std::mem::swap(&mut b_old, &mut b_new);
        }

        if r > 0 {
            let base = nb * w;
            let out_rest = &mut out[base..];
            let mut acc = id;
            for (i, (o, b)) in out_rest.iter_mut().zip(&b_old[1..]).enumerate() {
                acc = combine_el(acc, get(base + i));
                o.write(combine_el(*b, acc));
            }
        }
    }
    // SAFETY: all `n` elements were written above (or `n == 0`).
    unsafe { ret.set_len(n) };
    ret
}
