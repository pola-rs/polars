use crate::trusted_len::PushUnchecked;
use num::Float;
use std::ops::AddAssign;

pub fn ewma_no_nulls<T, I>(vals: I, len: usize, window: usize) -> Vec<T>
where
    T: Float + AddAssign,
    I: IntoIterator<Item = T>,
{
    let mut iter = vals.into_iter();
    if iter.size_hint().0 == 0 {
        return vec![];
    }
    let two = T::one() + T::one();
    let alpha = two / T::from(window + 1).unwrap();
    let mut weight = T::one();
    let mut out = Vec::with_capacity(len);

    let first = iter.next().unwrap();
    out.push(first);
    let mut ewma_old = first;

    for (i, val) in iter.enumerate() {
        let i = i + 1;
        let one_sub_alpha = T::one() - alpha;
        weight += one_sub_alpha.powf(T::from(i).unwrap());
        ewma_old = ewma_old * (one_sub_alpha) + val;
        // Safety:
        // we allocated vals.len()
        unsafe { out.push_unchecked(ewma_old / weight) }
    }

    out
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ewma() {
        let vals = [2.0, 5.0, 3.0];
        let out = ewma_no_nulls(vals, 3, 3);
        let expected = [2.0, 4.0, 3.4285714285714284];
        assert_eq!(out, expected);
    }
}
