use crate::trusted_len::{PushUnchecked, TrustedLen};
use num::Float;
use std::fmt::Debug;
use std::ops::AddAssign;
// See:
// https://github.com/pola-rs/polars/issues/2148
// https://stackoverflow.com/a/51392341/6717054

pub fn ewma_no_nulls<T, I>(vals: I, alpha: T) -> Vec<T>
where
    T: Float + AddAssign,
    I: IntoIterator<Item = T>,
    I::IntoIter: TrustedLen,
{
    let mut iter = vals.into_iter();
    let len = iter.size_hint().1.unwrap();
    if len == 0 {
        return vec![];
    }
    let mut weight = T::one();
    let mut out = Vec::with_capacity(len);

    let first = iter.next().unwrap();
    out.push(first);
    let mut ewma_old = first;
    let one_sub_alpha = T::one() - alpha;

    for (i, val) in iter.enumerate() {
        let i = i + 1;
        weight += one_sub_alpha.powf(T::from(i).unwrap());
        ewma_old = ewma_old * (one_sub_alpha) + val;
        // Safety:
        // we allocated vals.len()
        unsafe { out.push_unchecked(ewma_old / weight) }
    }

    out
}

pub fn ewma_inf_hist_no_nulls<T, I>(vals: I, alpha: T) -> Vec<T>
where
    T: Float + AddAssign + Debug,
    I: IntoIterator<Item = T>,
    I::IntoIter: TrustedLen,
{
    let mut iter = vals.into_iter();
    let len = iter.size_hint().1.unwrap();
    if len == 0 {
        return vec![];
    }

    let mut out = Vec::with_capacity(len);
    let first = iter.next().unwrap();
    out.push(first);
    let one_sub_alpha = T::one() - alpha;

    for (i, val) in iter.enumerate() {
        let i = i + 1;

        // Safety:
        // we add first, so i - 1 always exits
        let output_val = val * alpha + unsafe { *out.get_unchecked(i - 1) } * one_sub_alpha;

        // Safety:
        // we allocated vals.len()
        unsafe { out.push_unchecked(output_val) }
    }

    out
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ewma() {
        let vals = [2.0, 5.0, 3.0];
        let out = ewma_no_nulls(vals.iter().copied(), 0.5);
        let expected = [2.0, 4.0, 3.4285714285714284];
        assert_eq!(out, expected);

        let vals = [2.0, 5.0, 3.0];
        let out = ewma_inf_hist_no_nulls(vals.iter().copied(), 0.5);
        let expected = [2.0, 3.5, 3.25];
        assert_eq!(out, expected);
    }
}
