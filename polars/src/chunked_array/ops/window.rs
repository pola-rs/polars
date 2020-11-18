use crate::prelude::*;
use num::{Bounded, NumCast, Zero};
use std::ops::{Add, Div, Mul};

/// a fold function to compute the sum. Returns a Null if there is a single null in the window
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

/// a fold function to compute the sum. The null values are ignored.
fn sum_fold_ignore_null<T>(acc: Option<T>, opt_v: Option<T>) -> Option<T>
where
    T: Add<Output = T> + Copy,
{
    match acc {
        None => opt_v,
        Some(acc) => match opt_v {
            None => Some(acc),
            Some(v) => Some(acc + v),
        },
    }
}

/// a fold function to compute the minimum. Returns a Null if there is a single null in the window
fn min_fold<T>(acc: Option<T>, opt_v: Option<T>) -> Option<T>
where
    T: PartialOrd,
{
    match acc {
        None => None,
        Some(acc) => match opt_v {
            None => None,
            Some(v) => Some(if acc < v { acc } else { v }),
        },
    }
}

/// a fold function to compute the min. The null values are ignored.
fn min_fold_ignore_null<T>(acc: Option<T>, opt_v: Option<T>) -> Option<T>
where
    T: PartialOrd,
{
    match acc {
        None => opt_v,
        Some(acc) => match opt_v {
            None => Some(acc),
            Some(v) => Some(if acc < v { acc } else { v }),
        },
    }
}
/// a fold function to compute the maximum. Returns a Null if there is a single null in the window
fn max_fold<T>(acc: Option<T>, opt_v: Option<T>) -> Option<T>
where
    T: PartialOrd,
{
    match acc {
        None => None,
        Some(acc) => match opt_v {
            None => None,
            Some(v) => Some(if acc > v { acc } else { v }),
        },
    }
}

/// a fold function to compute the max. The null values are ignored.
fn max_fold_ignore_null<T>(acc: Option<T>, opt_v: Option<T>) -> Option<T>
where
    T: PartialOrd,
{
    match acc {
        None => opt_v,
        Some(acc) => match opt_v {
            None => Some(acc),
            Some(v) => Some(if acc > v { acc } else { v }),
        },
    }
}

fn rescale_window<T>(window: &[Option<T>], weight: &[T]) -> Vec<Option<T>>
where
    T: Mul<Output = T> + Copy,
{
    window
        .iter()
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

/// Apply weight to the current window and accumulate with a `fold_fn`.
fn apply_window<T, F>(
    weight: Option<&[T]>,
    window: &[Option<T>],
    fold_fn: F,
    init_fold: InitFold,
) -> Option<T>
where
    T: Copy + Add<Output = T> + Zero + Mul<Output = T> + Bounded,
    F: Fn(Option<T>, Option<T>) -> Option<T>,
{
    let init = match init_fold {
        InitFold::Zero => Zero::zero(),
        InitFold::Min => Bounded::min_value(),
        InitFold::Max => Bounded::max_value(),
    };

    match weight {
        None => window.iter().copied().fold(Some(init), fold_fn),
        Some(weight) => rescale_window(window, weight)
            .into_iter()
            .fold(Some(init), fold_fn),
    }
}

/// Cast weights of f64 to T::Native
fn weight_to_native<Native: NumCast>(weight: &[f64]) -> Vec<Native> {
    weight
        .iter()
        .map(|&v| NumCast::from(v).expect("all numeric types are castable"))
        .collect()
}

fn finish_rolling_method<T, F>(
    ca: &ChunkedArray<T>,
    fold_fn: F,
    window_size: usize,
    weight: Option<&[f64]>,
    init_fold: InitFold,
) -> Result<ChunkedArray<T>>
where
    T: PolarsNumericType,
    T::Native: Zero
        + Bounded
        + NumCast
        + Div<Output = T::Native>
        + Mul<Output = T::Native>
        + PartialOrd
        + Copy,
    F: Fn(Option<T::Native>, Option<T::Native>) -> Option<T::Native> + Copy,
{
    let weight: Option<Vec<T::Native>> = weight.map(weight_to_native);
    let window = vec![None; window_size];
    let mut idx_count = 0;
    let ca = if ca.null_count() == 0 {
        ca.into_no_null_iter()
            .scan((window, 0usize), |state, v| {
                idx_count = update_state(state, idx_count, Some(v), window_size);
                let (window, _) = state;
                let sum = apply_window(weight.as_deref(), window, fold_fn, init_fold);
                Some(sum)
            })
            .collect()
    } else {
        ca.into_iter()
            .scan((window, 0usize), |state, opt_v| {
                idx_count = update_state(state, idx_count, opt_v, window_size);
                let (window, _) = state;
                Some(apply_window(weight.as_deref(), window, fold_fn, init_fold))
            })
            .collect()
    };
    Ok(ca)
}

#[derive(Clone, Copy)]
pub enum InitFold {
    Zero,
    Max,
    Min,
}

impl<T> ChunkWindow<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Zero
        + Bounded
        + NumCast
        + Div<Output = T::Native>
        + Mul<Output = T::Native>
        + PartialOrd
        + Copy,
{
    fn rolling_sum(
        &self,
        window_size: usize,
        weight: Option<&[f64]>,
        ignore_null: bool,
    ) -> Result<Self> {
        let fold_fn = if ignore_null {
            sum_fold_ignore_null::<T::Native>
        } else {
            sum_fold::<T::Native>
        };

        finish_rolling_method(self, fold_fn, window_size, weight, InitFold::Zero)
    }

    fn rolling_mean(
        &self,
        window_size: usize,
        weight: Option<&[f64]>,
        ignore_null: bool,
    ) -> Result<Self> {
        let ca = self.rolling_sum(window_size, weight, ignore_null)?;
        Ok(&ca / window_size)
    }

    fn rolling_min(
        &self,
        window_size: usize,
        weight: Option<&[f64]>,
        ignore_null: bool,
    ) -> Result<Self> {
        let fold_fn = if ignore_null {
            min_fold_ignore_null::<T::Native>
        } else {
            min_fold::<T::Native>
        };

        finish_rolling_method(self, fold_fn, window_size, weight, InitFold::Max)
    }

    fn rolling_max(
        &self,
        window_size: usize,
        weight: Option<&[f64]>,
        ignore_null: bool,
    ) -> Result<Self> {
        let fold_fn = if ignore_null {
            max_fold_ignore_null::<T::Native>
        } else {
            max_fold::<T::Native>
        };

        finish_rolling_method(self, fold_fn, window_size, weight, InitFold::Min)
    }

    fn rolling_custom<F>(
        &self,
        window_size: usize,
        weight: Option<&[f64]>,
        fold_fn: F,
        init_fold: InitFold,
    ) -> Result<Self>
    where
        F: Fn(Option<T::Native>, Option<T::Native>) -> Option<T::Native> + Copy,
    {
        finish_rolling_method(self, fold_fn, window_size, weight, init_fold)
    }
}

impl ChunkWindow<u8> for ListChunked {}
impl ChunkWindow<u8> for Utf8Chunked {}
impl ChunkWindow<u8> for BooleanChunked {}
impl<T> ChunkWindow<u8> for ObjectChunked<T> {}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_rolling() {
        let ca = Int32Chunked::new_from_slice("foo", &[1, 2, 3, 2, 1]);
        let a = ca.rolling_sum(2, None, true).unwrap();
        assert_eq!(
            Vec::from(&a),
            [1, 3, 5, 5, 3]
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>()
        );
        let a = ca.rolling_min(2, None, true).unwrap();
        assert_eq!(
            Vec::from(&a),
            [1, 1, 2, 2, 1]
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>()
        );
        let a = ca
            .rolling_max(2, Some(&[1., 1., 1., 1., 1.]), true)
            .unwrap();
        assert_eq!(
            Vec::from(&a),
            [1, 2, 3, 3, 2]
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>()
        );
    }
}
