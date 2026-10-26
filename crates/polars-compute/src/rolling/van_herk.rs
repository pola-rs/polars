use polars_utils::min_max::MinMaxPolicy;

use super::*;

pub(super) fn rolling_minmax_shifted<T, P>(values: &[T], w: usize, shift: usize) -> Vec<T>
where
    T: NativeType + IsFloat + Bounded,
    P: MinMaxPolicy,
{
    let mut out = van_herk_kernel::<false, T, P>(values, &Bitmap::new(), w);
    if shift > 0 {
        shift_trailing::<false, T, P>(&mut out, values, &Bitmap::new(), w, shift);
    }
    out
}

/// Shift trailing output with a backward suffix scan.
pub(super) fn shift_trailing<const MASKED: bool, T, P>(
    out: &mut [T],
    values: &[T],
    validity: &Bitmap,
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
    let id = P::identity();
    let get = |i: usize| {
        if MASKED {
            // SAFETY: callers keep `i < n` and `validity` covers `values`.
            if unsafe { !validity.get_bit_unchecked(i) } {
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
            acc = P::best(get(i), acc);
        }
        for p in (n.saturating_sub(w - 1)..hi).rev() {
            acc = P::best(get(p), acc);
            out[p + w - 1 - shift] = acc;
        }
    } else {
        for i in (0..n).rev() {
            acc = P::best(get(i), acc);
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
    validity: &Bitmap,
    w: usize,
) -> Vec<T>
where
    T: NativeType + IsFloat + Bounded,
    P: MinMaxPolicy,
{
    let id = P::identity();
    let n = values.len();
    let get = |i: usize| {
        if MASKED {
            // SAFETY: callers keep `i < n` and `validity` covers `values`.
            if unsafe { !validity.get_bit_unchecked(i) } {
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
            acc = P::best(acc, get(i));
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
                prefix = P::best(prefix, vf);
                suffix = P::best(vb, suffix);
                out_block[j].write(P::best(b_old[j + 1], prefix));
                b_new[w - 1 - j] = suffix;
            }
            prefix = P::best(prefix, get(base + w - 1));
            suffix = P::best(get(base), suffix);
            out_block[w - 1].write(prefix);
            b_new[0] = suffix;
            std::mem::swap(&mut b_old, &mut b_new);
        }

        if r > 0 {
            let base = nb * w;
            let out_rest = &mut out[base..];
            let mut acc = id;
            for (i, (o, b)) in out_rest.iter_mut().zip(&b_old[1..]).enumerate() {
                acc = P::best(acc, get(base + i));
                o.write(P::best(*b, acc));
            }
        }
    }
    // SAFETY: all `n` elements were written above (or `n == 0`).
    unsafe { ret.set_len(n) };
    ret
}
