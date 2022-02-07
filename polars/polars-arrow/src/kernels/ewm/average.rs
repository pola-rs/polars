use crate::trusted_len::{PushUnchecked, TrustedLen};
use num::Float;
use std::fmt::Debug;
use std::ops::AddAssign;
// See:
// https://github.com/pola-rs/polars/issues/2148
// https://stackoverflow.com/a/51392341/6717054

// this is the adjusted variant in pandas
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

    let mut i = T::from(out.len()).unwrap();
    for val in iter {
        weight += one_sub_alpha.powf(i);
        ewma_old = ewma_old * (one_sub_alpha) + val;
        // Safety:
        // we allocated vals.len()
        unsafe { out.push_unchecked(ewma_old / weight) };
        i += T::one();
    }

    out
}

pub fn ewma<T, I>(vals: I, alpha: T) -> (usize, Vec<T>)
where
    T: Float + AddAssign,
    I: IntoIterator<Item = Option<T>>,
    I::IntoIter: TrustedLen,
{
    let mut iter = vals.into_iter();
    let len = iter.size_hint().1.unwrap();
    if len == 0 {
        return (0, vec![]);
    }
    let mut weight = T::one();
    let mut out = Vec::with_capacity(len);

    let leading_null_count = set_first_none_null(&mut iter, &mut out);
    let mut ewma_old = out[out.len() - 1];
    let one_sub_alpha = T::one() - alpha;

    let mut i = T::one();
    let mut prev = out[out.len() - 1];
    for opt_val in iter {
        prev = match opt_val {
            Some(val) => {
                weight += one_sub_alpha.powf(i);
                ewma_old = ewma_old * (one_sub_alpha) + val;
                i += T::one();
                ewma_old / weight
            }
            None => prev,
        };
        // Safety:
        // we allocated vals.len()
        unsafe { out.push_unchecked(prev) };
    }

    (leading_null_count, out)
}

// this is the non-adjusted variant in pandas
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

    let mut prev = out[0];
    for val in iter {
        let output_val = val * alpha + prev * one_sub_alpha;
        prev = output_val;

        // Safety:
        // we allocated vals.len()
        unsafe { out.push_unchecked(output_val) }
    }

    out
}

// this is the non-adjusted variant in pandas
/// # Arguments
///
/// * `vals` - Iterator of optional values
/// * `alpha` - Smoothing factor
///
/// Returns the a tuple with:
/// * `leading_null_count` - the amount of nulls that must be set by the caller
/// * `smoothed values` - The result of the ewma
///
pub fn ewma_inf_hists<T, I>(vals: I, alpha: T) -> (usize, Vec<T>)
where
    T: Float + AddAssign + Debug,
    I: IntoIterator<Item = Option<T>>,
    I::IntoIter: TrustedLen,
{
    let mut iter = vals.into_iter();
    let len = iter.size_hint().1.unwrap();
    if len == 0 {
        return (0, vec![]);
    }

    let mut out = Vec::with_capacity(len);

    let leading_null_count = set_first_none_null(&mut iter, &mut out);
    let one_sub_alpha = T::one() - alpha;
    let mut prev = out[out.len() - 1];

    for opt_val in iter {
        let output_val = match opt_val {
            Some(val) => {
                // Safety:
                // we add first, so i - 1 always exits
                let output = val * alpha + prev * one_sub_alpha;
                prev = output;
                prev
            }
            None => prev,
        };

        // Safety:
        // we allocated vals.len()
        unsafe { out.push_unchecked(output_val) }
    }

    (leading_null_count, out)
}

pub fn set_first_none_null<T, I>(iter: &mut I, out: &mut Vec<T>) -> usize
where
    T: Float + AddAssign,
    I: Iterator<Item = Option<T>>,
{
    let mut leading_null_count = 0;
    // find first non null
    for opt_val in iter {
        match opt_val {
            // these will be later masked out by the validity
            None => {
                leading_null_count += 1;
                unsafe { out.push_unchecked(T::zero()) };
            }
            Some(val) => {
                unsafe { out.push_unchecked(val) };
                break;
            }
        }
    }
    leading_null_count
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

    #[test]
    fn test_ewma_null() {
        let vals = &[
            Some(2.0),
            Some(3.0),
            Some(5.0),
            Some(7.0),
            None,
            None,
            None,
            Some(4.0),
        ];
        let (cnt, out) = ewma_inf_hists(vals.into_iter().copied(), 0.5);
        assert_eq!(cnt, 0);
        let expected = [2.0, 2.5, 3.75, 5.375, 5.375, 5.375, 5.375, 4.6875];
        assert_eq!(out, expected);
        let vals = &[
            None,
            None,
            Some(5.0),
            Some(7.0),
            None,
            Some(2.0),
            Some(1.0),
            Some(4.0),
        ];
        let (cnt, out) = ewma_inf_hists(vals.into_iter().copied(), 0.5);
        let expected = [0.0, 0.0, 5.0, 6.0, 6.0, 4.0, 2.5, 3.25];
        assert_eq!(cnt, 2);
        assert_eq!(out, expected);

        let (cnt, out) = ewma(vals.into_iter().copied(), 0.5);
        let expected = [
            0.0,
            0.0,
            5.0,
            6.333333333333333,
            6.333333333333333,
            3.857142857142857,
            2.3333333333333335,
            3.193548387096774,
        ];
        assert_eq!(cnt, 2);
        assert_eq!(out, expected);

        let vals = &[
            None,
            Some(1.0),
            Some(5.0),
            Some(7.0),
            None,
            Some(2.0),
            Some(1.0),
            Some(4.0),
        ];
        let (cnt, out) = ewma(vals.into_iter().copied(), 0.5);
        let expected = [
            0.0,
            1.0,
            3.6666666666666665,
            5.571428571428571,
            5.571428571428571,
            3.6666666666666665,
            2.2903225806451615,
            3.1587301587301586,
        ];
        assert_eq!(cnt, 1);
        assert_eq!(out, expected);
    }
}
