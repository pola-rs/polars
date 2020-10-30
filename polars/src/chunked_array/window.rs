use crate::prelude::*;
use num::{NumCast, Zero};
use std::ops::{Add, Div, Mul};

fn sum_fold<T>(acc: Option<T>, opt_v: Option<T>) -> Option<T>
where
    T: Add<Output = T> + Copy,
{
    match acc {
        None => None,
        Some(acc) => match opt_v {
            None => None,
            Some(v) => Some(acc + v),
        },
    }
}

fn rescale_window<T>(window: &[Option<T>], weight: &[T]) -> Vec<Option<T>>
where
    T: Mul<Output = T> + Copy,
{
    window
        .into_iter()
        .zip(weight)
        .map(|(opt_a, &b)| opt_a.map(|a| a * b))
        .collect()
}

fn update_state<T>(
    state: &mut (Vec<Option<T>>, usize),
    idx_count: usize,
    opt_v: Option<T>,
    window_size: usize,
) -> usize {
    let (window, idx) = state;
    let old_value = &mut window[*idx];
    let mut new_val = opt_v;
    std::mem::swap(old_value, &mut new_val);
    let idx_count = idx_count + 1;
    state.1 = idx_count % window_size;
    idx_count
}

fn apply_window<T>(weight: Option<&[T]>, window: &[Option<T>]) -> Option<T>
where
    T: Copy + Add<Output = T> + Zero + Mul<Output = T>,
{
    match weight {
        None => window.iter().copied().fold(Some(Zero::zero()), sum_fold),
        Some(weight) => rescale_window(window, weight)
            .into_iter()
            .fold(Some(Zero::zero()), sum_fold),
    }
}

impl<T> ChunkWindow<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Zero + NumCast + Div<Output = T::Native> + Mul<Output = T::Native> + Copy,
{
    fn rolling_sum(&self, window_size: usize, weight: Option<&[T::Native]>) -> Result<Self> {
        let mut window = Vec::with_capacity(window_size);
        for _ in 0..window_size {
            window.push(None)
        }
        let mut idx_count = 0;
        // we create a window array of size window size.
        // and we have a rolling index that points to the oldest value in this window.
        // the index determines which value to swap at every iteration
        let ca = if self.null_count() == 0 {
            self.into_no_null_iter()
                .scan((window, 0usize), |state, v| {
                    idx_count = update_state(state, idx_count, Some(v), window_size);
                    let (window, _) = state;
                    let sum = apply_window(weight, window);
                    Some(sum)
                })
                .collect()
        } else {
            self.into_iter()
                .scan((window, 0usize), |state, opt_v| {
                    idx_count = update_state(state, idx_count, opt_v, window_size);
                    let (window, _) = state;
                    Some(apply_window(weight, window))
                })
                .collect()
        };
        Ok(ca)
    }
    fn rolling_mean(&self, window_size: usize, weight: Option<&[T::Native]>) -> Result<Self> {
        let ca = self.rolling_sum(window_size, weight)?;
        Ok(&ca / window_size)
    }
}

impl ChunkWindow<u8> for ListChunked {}
impl ChunkWindow<u8> for Utf8Chunked {}
impl ChunkWindow<u8> for BooleanChunked {}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_rolling() {
        let ca =
            Int32Chunked::new_from_aligned_vec("foo", (0..15).into_iter().map(|v| v % 3).collect());
        let a = ca.rolling_sum(5, None).unwrap();
        dbg!(a);
        let a = ca.rolling_mean(5, Some(&[1, 2, 1, 1, 3])).unwrap();
        dbg!(a);
    }
}
