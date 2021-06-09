use crate::prelude::*;
use num::{Bounded, NumCast, One, Zero};
use std::ops::{Add, Div, Mul, Rem, Sub};

/// a fold function to compute the sum. Returns a Null if there is a single null in the window
fn sum_fold<T>(acc: Option<T>, opt_v: Option<T>) -> Option<T>
where
    T: Add<Output = T> + Copy,
{
    match acc {
        None => None,
        Some(acc) => opt_v.map(|v| acc + v),
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
        Some(acc) => opt_v.map(|v| if acc < v { acc } else { v }),
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
        Some(acc) => opt_v.map(|v| if acc > v { acc } else { v }),
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

/// a fold function to compute the window size. The null values are ignored.
fn window_size_fold_ignore_null<T>(acc: Option<T>, opt_v: Option<T>) -> Option<T>
where
    T: Add<Output = T> + Copy + One,
{
    match acc {
        None => Some(T::one()),
        Some(acc) => match opt_v {
            None => Some(acc),
            _ => Some(acc + T::one()),
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

// the state holds the window and the current idx of the operation.
// The value at the end of the window is not always the latest value (i.e. the value at the idx)
// otherwise we have to move all the values if we push one to the window.
// Instead we use a ring buffer
// Therefore the oldest value in the window gets replaced by the new value and is determined by:
// current_idx % window_size
//
//
// the latest state values is the amount of Some<T> values in the window
fn update_state<T>(
    // (window , oldest_value_idx, amount_Some)
    state: &mut (Vec<Option<T>>, u32, u32),
    // count of the loop
    idx_count: u32,
    // new value
    opt_v: Option<T>,
    // size of the window
    window_size: u32,
) -> u32 {
    let (window, idx, _) = state;
    let old_value = &mut window[*idx as usize];
    let mut new_val = opt_v;

    if new_val.is_some() {
        state.2 += 1;
    }
    if old_value.is_some() {
        state.2 -= 1;
    }

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
    window_size: u32,
    weight: Option<&[f64]>,
    init_fold: InitFold,
    min_periods: u32,
) -> ChunkedArray<T>
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
    let window = vec![None; window_size as usize];
    let mut idx_count = 0;
    if ca.null_count() == 0 {
        ca.into_no_null_iter()
            .scan((window, 0u32, 0u32), |state, v| {
                idx_count = update_state(state, idx_count, Some(v), window_size);
                let (window, _, some_count) = state;
                if *some_count < min_periods {
                    Some(None)
                } else {
                    let sum = apply_window(weight.as_deref(), window, fold_fn, init_fold);
                    Some(sum)
                }
            })
            .collect()
    } else {
        ca.into_iter()
            .scan((window, 0u32, 0u32), |state, opt_v| {
                idx_count = update_state(state, idx_count, opt_v, window_size);
                let (window, _, some_count) = state;
                if *some_count < min_periods {
                    Some(None)
                } else {
                    Some(apply_window(weight.as_deref(), window, fold_fn, init_fold))
                }
            })
            .collect()
    }
}

#[derive(Clone, Copy)]
pub enum InitFold {
    Zero,
    Max,
    Min,
}

impl<T> ChunkWindow for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + Rem<Output = T::Native>
        + Zero
        + Bounded
        + NumCast
        + PartialOrd
        + One
        + Copy,
{
    fn rolling_sum(
        &self,
        window_size: u32,
        weight: Option<&[f64]>,
        ignore_null: bool,
        min_periods: u32,
    ) -> Result<Self> {
        check_input(window_size, min_periods)?;
        let fold_fn = if ignore_null {
            sum_fold_ignore_null::<T::Native>
        } else {
            sum_fold::<T::Native>
        };

        Ok(finish_rolling_method(
            self,
            fold_fn,
            window_size,
            weight,
            InitFold::Zero,
            min_periods,
        ))
    }

    fn rolling_mean(
        &self,
        window_size: u32,
        weight: Option<&[f64]>,
        ignore_null: bool,
        min_periods: u32,
    ) -> Result<Self> {
        check_input(window_size, min_periods)?;
        let rolling_window_size = self.window_size(window_size, None, min_periods);
        let ca = self.rolling_sum(window_size, weight, ignore_null, min_periods)?;
        Ok((&ca).div(&rolling_window_size))
    }

    fn rolling_min(
        &self,
        window_size: u32,
        weight: Option<&[f64]>,
        ignore_null: bool,
        min_periods: u32,
    ) -> Result<Self> {
        check_input(window_size, min_periods)?;
        let fold_fn = if ignore_null {
            min_fold_ignore_null::<T::Native>
        } else {
            min_fold::<T::Native>
        };

        Ok(finish_rolling_method(
            self,
            fold_fn,
            window_size,
            weight,
            InitFold::Max,
            min_periods,
        ))
    }

    fn rolling_max(
        &self,
        window_size: u32,
        weight: Option<&[f64]>,
        ignore_null: bool,
        min_periods: u32,
    ) -> Result<Self> {
        check_input(window_size, min_periods)?;
        let fold_fn = if ignore_null {
            max_fold_ignore_null::<T::Native>
        } else {
            max_fold::<T::Native>
        };

        Ok(finish_rolling_method(
            self,
            fold_fn,
            window_size,
            weight,
            InitFold::Min,
            min_periods,
        ))
    }
}

impl<T> ChunkWindowCustom<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Zero
        + Bounded
        + NumCast
        + Div<Output = T::Native>
        + Mul<Output = T::Native>
        + PartialOrd
        + One
        + Copy,
{
    fn rolling_custom<F>(
        &self,
        window_size: u32,
        weight: Option<&[f64]>,
        fold_fn: F,
        init_fold: InitFold,
        min_periods: u32,
    ) -> Result<Self>
    where
        F: Fn(Option<T::Native>, Option<T::Native>) -> Option<T::Native> + Copy,
    {
        Ok(finish_rolling_method(
            self,
            fold_fn,
            window_size,
            weight,
            init_fold,
            min_periods,
        ))
    }
}

/// utility
fn check_input(window_size: u32, min_periods: u32) -> Result<()> {
    if min_periods > window_size {
        Err(PolarsError::ValueError(
            "`windows_size` should be >= `min_periods`".into(),
        ))
    } else {
        Ok(())
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Zero
        + Bounded
        + NumCast
        + Div<Output = T::Native>
        + Mul<Output = T::Native>
        + PartialOrd
        + One
        + Copy,
{
    /// Compute the window size during traversion of the array.
    /// The window size may be less than `window_size` at the edges, or when null values should be ignored.
    fn window_size(&self, window_size: u32, weight: Option<&[f64]>, min_periods: u32) -> Self {
        let fold_fn = window_size_fold_ignore_null::<T::Native>;
        let init_fold = InitFold::Zero;

        finish_rolling_method(self, fold_fn, window_size, weight, init_fold, min_periods)
    }
}

impl ChunkWindow for ListChunked {}
impl ChunkWindow for Utf8Chunked {}
impl ChunkWindow for BooleanChunked {}
impl ChunkWindow for CategoricalChunked {}
#[cfg(feature = "object")]
impl<T> ChunkWindow for ObjectChunked<T> {}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_rolling() {
        let ca = Int32Chunked::new_from_slice("foo", &[1, 2, 3, 2, 1]);
        let a = ca.rolling_sum(2, None, true, 0).unwrap();
        assert_eq!(
            Vec::from(&a),
            [1, 3, 5, 5, 3]
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>()
        );
        let a = ca.rolling_min(2, None, true, 0).unwrap();
        assert_eq!(
            Vec::from(&a),
            [1, 1, 2, 2, 1]
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>()
        );
        let a = ca
            .rolling_max(2, Some(&[1., 1., 1., 1., 1.]), true, 0)
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

    #[test]
    fn test_rolling_min_periods() {
        let ca = Int32Chunked::new_from_slice("foo", &[1, 2, 3, 2, 1]);
        let a = ca.rolling_max(2, None, true, 2).unwrap();
        assert_eq!(Vec::from(&a), &[None, Some(2), Some(3), Some(3), Some(2)]);
    }

    #[test]
    fn test_rolling_mean() {
        let ca = Float64Chunked::new_from_opt_slice(
            "foo",
            &[
                Some(0.0),
                Some(1.0),
                Some(2.0),
                None,
                None,
                Some(5.0),
                Some(6.0),
            ],
        );

        // check err on wrong input
        assert!(ca.rolling_mean(1, None, true, 2).is_err());

        // validate that we divide by the proper window length. (same as pandas)
        let a = ca.rolling_mean(3, None, true, 1).unwrap();
        assert_eq!(
            Vec::from(&a),
            &[
                Some(0.0),
                Some(0.5),
                Some(1.0),
                Some(1.5),
                Some(2.0),
                Some(5.0),
                Some(5.5)
            ]
        );
    }
}
