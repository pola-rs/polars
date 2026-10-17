use polars_utils::min_max::MinMaxPolicy;

use super::*;

pub(super) fn rolling_minmax_centered<T, P>(values: &[T], w: usize, shift: usize) -> Vec<T>
where
    T: NativeType + IsFloat + Bounded,
    P: MinMaxPolicy,
{
    let mut out = van_herk_kernel::<false, T, P>(values, &[], w);
    if shift > 0 {
        recenter_trailing::<false, T, P>(&mut out, values, &[], w, shift);
    }
    out
}

/// Fold identity, including infinities for floating-point inputs.
#[inline]
fn identity<T: NativeType + IsFloat + Bounded, P: MinMaxPolicy>() -> T {
    let (lo, hi) = if T::is_float() {
        (T::neg_inf_value(), T::pos_inf_value())
    } else {
        (T::min_value(), T::max_value())
    };
    if P::is_better(&lo, &hi) { hi } else { lo }
}

/// Stable combine: ties keep the earlier NaN payload or signed zero.
#[inline(always)]
fn combine<T: NativeType, P: MinMaxPolicy>(earlier: T, later: T) -> T {
    if P::is_better(&later, &earlier) {
        later
    } else {
        earlier
    }
}

/// Recenter trailing output with a backward suffix scan.
pub(super) fn recenter_trailing<const MASKED: bool, T, P>(
    out: &mut [T],
    values: &[T],
    validity_words: &[u64],
    w: usize,
    shift: usize,
) where
    T: NativeType + IsFloat + Bounded,
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

/// Shared block scan; the `MASKED` constant specializes the null checks away for
/// non-null input.
pub(super) fn van_herk_kernel<const MASKED: bool, T, P>(
    values: &[T],
    validity_words: &[u64],
    w: usize,
) -> Vec<T>
where
    T: NativeType + IsFloat + Bounded,
    P: MinMaxPolicy,
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
            acc = combine::<T, P>(acc, get(i));
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
                prefix = combine::<T, P>(prefix, vf);
                // Right-to-left keeps the leftmost tie or NaN.
                suffix = combine::<T, P>(vb, suffix);
                out_block[j].write(combine::<T, P>(b_old[j + 1], prefix));
                b_new[w - 1 - j] = suffix;
            }
            prefix = combine::<T, P>(prefix, get(base + w - 1));
            suffix = combine::<T, P>(get(base), suffix);
            out_block[w - 1].write(prefix);
            b_new[0] = suffix;
            std::mem::swap(&mut b_old, &mut b_new);
        }

        if r > 0 {
            let base = nb * w;
            let out_rest = &mut out[base..];
            let mut acc = id;
            for (i, (o, b)) in out_rest.iter_mut().zip(&b_old[1..]).enumerate() {
                acc = combine::<T, P>(acc, get(base + i));
                o.write(combine::<T, P>(*b, acc));
            }
        }
    }
    // SAFETY: all `n` elements were written above (or `n == 0`).
    unsafe { ret.set_len(n) };
    ret
}
