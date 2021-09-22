use crate::prelude::*;
use arrow::array::{Array, PrimitiveArray};
use arrow::bitmap::utils::count_zeros;
use arrow::bitmap::MutableBitmap;
use num::{Bounded, Float, NumCast, One, Zero};
use polars_arrow::bit_util::unset_bit_raw;
use polars_arrow::trusted_len::PushUnchecked;
use polars_arrow::utils::CustomIterTools;
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
    // new value
    opt_v: Option<T>,
    // size of the window
    window_size: u32,
) {
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

    // this removes an expensive modulo
    state.1 += 1;
    if state.1 == window_size {
        state.1 = 0
    }
}

/// Apply weight to the current window and accumulate with a `fold_fn`.
fn apply_window<T, F>(weight: Option<&[T]>, window: &[Option<T>], fold_fn: F, init: T) -> Option<T>
where
    T: Copy + Add<Output = T> + Zero + Mul<Output = T> + Bounded,
    F: Fn(Option<T>, Option<T>) -> Option<T>,
{
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

    let init = match init_fold {
        InitFold::Zero => Zero::zero(),
        InitFold::Min => Bounded::min_value(),
        InitFold::Max => Bounded::max_value(),
    };

    if ca.null_count() == 0 {
        ca.into_no_null_iter()
            .scan((window, 0u32, 0u32), |state, v| {
                update_state(state, Some(v), window_size);
                let (window, _, some_count) = state;
                if *some_count < min_periods {
                    Some(None)
                } else {
                    let sum = apply_window(weight.as_deref(), window, fold_fn, init);
                    Some(sum)
                }
            })
            .trust_my_length(ca.len())
            .collect_trusted()
    } else {
        ca.into_iter()
            .scan((window, 0u32, 0u32), |state, opt_v| {
                update_state(state, opt_v, window_size);
                let (window, _, some_count) = state;
                if *some_count < min_periods {
                    Some(None)
                } else {
                    Some(apply_window(weight.as_deref(), window, fold_fn, init))
                }
            })
            .trust_my_length(ca.len())
            .collect_trusted()
    }
}

impl<T> ChunkWindowMean for ChunkedArray<T>
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
    ChunkedArray<T>: IntoSeries,
{
    fn rolling_mean(
        &self,
        window_size: u32,
        weight: Option<&[f64]>,
        ignore_null: bool,
        min_periods: u32,
    ) -> Result<Series> {
        match self.dtype() {
            DataType::Float32 | DataType::Float64 => {
                check_input(window_size, min_periods)?;
                let ca = self.rolling_sum(window_size, weight, ignore_null, min_periods)?;
                let rolling_window_size = self.window_size(window_size, None, min_periods);
                Ok((&ca).div(&rolling_window_size).into_series())
            }
            _ => {
                let ca = self.cast::<Float64Type>()?;
                ca.rolling_mean(window_size, weight, ignore_null, min_periods)
            }
        }
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

/// This is similar to how rolling_min, sum, max, mean
/// is implemented. It takes a window, weights it and applies a fold aggregator
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

impl<T> ChunkRollApply for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Zero,
    Self: IntoSeries,
{
    fn rolling_apply(&self, window_size: usize, f: &dyn Fn(&Series) -> Series) -> Result<Self> {
        if window_size >= self.len() {
            return Ok(Self::full_null(self.name(), self.len()));
        }
        let ca = self.rechunk();
        let arr = ca.downcast_iter().next().unwrap();

        let series_container =
            ChunkedArray::<T>::new_from_slice("", &[T::Native::zero()]).into_series();
        let array_ptr = &series_container.chunks()[0];
        let ptr = Arc::as_ptr(array_ptr) as *mut dyn Array as *mut PrimitiveArray<T::Native>;
        let mut builder = PrimitiveChunkedBuilder::<T>::new(self.name(), self.len());
        for _ in 0..window_size - 1 {
            builder.append_null();
        }

        for offset in 0..self.len() + 1 - window_size {
            let arr_window = arr.slice(offset, window_size);

            // Safety.
            // ptr is not dropped as we are in scope
            // We are also the only owner of the contents of the Arc
            // we do this to reduce heap allocs.
            unsafe {
                *ptr = arr_window;
            }

            let s = f(&series_container);
            let out = self.unpack_series_matching_type(&s)?;
            builder.append_option(out.get(0));
        }

        Ok(builder.finish())
    }
}

fn variance<T>(vals: &[T]) -> T
where
    T: Float + std::iter::Sum,
{
    let len = T::from(vals.len()).unwrap();
    let mean = vals.iter().copied().sum::<T>() / len;

    let mut sum = T::zero();
    for &val in vals {
        let v = val - mean;
        sum = sum + v * v
    }
    sum / (len - T::one())
}

impl<T> ChunkedArray<T>
where
    ChunkedArray<T>: IntoSeries,
    T: PolarsFloatType,
    T::Native: Default + std::iter::Sum + Float,
{
    pub fn rolling_apply_float<F>(&self, window_size: usize, f: F) -> Result<Self>
    where
        F: Fn(&ChunkedArray<T>) -> Option<T::Native>,
        T::Native: Zero,
    {
        if window_size >= self.len() {
            return Ok(Self::full_null(self.name(), self.len()));
        }
        let ca = self.rechunk();
        let arr = ca.downcast_iter().next().unwrap();

        let arr_container = ChunkedArray::<T>::new_from_slice("", &[T::Native::zero()]);
        let array_ptr = &arr_container.chunks()[0];
        let ptr = Arc::as_ptr(array_ptr) as *mut dyn Array as *mut PrimitiveArray<T::Native>;

        let mut validity = MutableBitmap::with_capacity(ca.len());
        validity.extend_constant(window_size - 1, false);
        validity.extend_constant(ca.len() - (window_size - 1), true);
        let validity_ptr = validity.as_slice().as_ptr() as *mut u8;

        let mut values = AlignedVec::with_capacity(ca.len());
        values.extend_constant(window_size - 1, Default::default());

        for offset in 0..self.len() + 1 - window_size {
            let arr_window = arr.slice(offset, window_size);

            // Safety.
            // ptr is not dropped as we are in scope
            // We are also the only owner of the contents of the Arc
            // we do this to reduce heap allocs.
            unsafe {
                *ptr = arr_window;
            }

            let out = f(&arr_container);
            match out {
                Some(v) => unsafe { values.push_unchecked(v) },
                None => unsafe { unset_bit_raw(validity_ptr, offset + window_size - 1) },
            }
        }
        let arr = PrimitiveArray::from_data(
            T::get_dtype().to_arrow(),
            values.into(),
            Some(validity.into()),
        );
        Ok(Self::new_from_chunks(self.name(), vec![Arc::new(arr)]))
    }

    pub fn rolling_var(&self, window_size: usize) -> Series {
        if window_size >= self.len() {
            return Self::full_null(self.name(), self.len()).into();
        }

        let ca = self.rechunk();
        let arr = ca.downcast_iter().next().unwrap();
        let values = arr.values().as_slice();

        let mut validity = MutableBitmap::with_capacity(ca.len());
        validity.extend_constant(window_size - 1, false);
        validity.extend_constant(ca.len() - (window_size - 1), true);
        let validity_ptr = validity.as_slice().as_ptr() as *mut u8;

        let mut rolling_values = AlignedVec::with_capacity(ca.len());
        rolling_values.extend_constant(window_size - 1, Default::default());

        if ca.null_count() == 0 {
            for offset in 0..self.len() + 1 - window_size {
                let window = &values[offset..offset + window_size];
                let val = variance(window);

                unsafe {
                    // Safety:
                    // We pre-allocated enough capacity
                    rolling_values.push_unchecked(val);
                };
            }
        } else {
            let old_validity = arr.validity().as_ref().unwrap().clone();
            let (bytes, bytes_offset, _) = old_validity.as_slice();
            for offset in 0..self.len() + 1 - window_size {
                if count_zeros(bytes, bytes_offset + offset, window_size) > 0 {
                    unsafe {
                        // Safety:
                        // We pre-allocated enough capacity
                        rolling_values.push_unchecked(Default::default());
                        // Safety:
                        // We are in bounds
                        unset_bit_raw(validity_ptr, offset + window_size - 1)
                    };
                } else {
                    let window = &values[offset..offset + window_size];
                    let val = variance(window);
                    // Safety:
                    // We pre-allocated enough capacity
                    unsafe { rolling_values.push_unchecked(val) };
                }
            }
        }

        let arr = PrimitiveArray::from_data(
            T::get_dtype().to_arrow(),
            rolling_values.into(),
            Some(validity.into()),
        );
        Self::new_from_chunks(self.name(), vec![Arc::new(arr)]).into()
    }

    pub fn rolling_std(&self, window_size: usize) -> Series {
        let s = self.rolling_var(window_size);
        // Safety:
        // We are still guarded by the type system.
        match self.dtype() {
            DataType::Float32 => s.f32().unwrap().pow_f32(0.5).into_series(),
            DataType::Float64 => s.f64().unwrap().pow_f64(0.5).into_series(),
            _ => unreachable!(),
        }
    }
}

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
        let a = a.f64().unwrap();
        assert_eq!(
            Vec::from(a),
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

        // integers
        let ca = Int32Chunked::new_from_slice("", &[1, 8, 6, 2, 16, 10]);
        let out = ca.rolling_mean(2, None, true, 2).unwrap();
        let out = out.f64().unwrap();
        assert_eq!(
            Vec::from(out),
            &[None, Some(4.5), Some(7.0), Some(4.0), Some(9.0), Some(13.0),]
        );
    }

    #[test]
    fn test_rolling_apply() {
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

        let out = ca.rolling_apply(3, &|s| s.sum_as_series()).unwrap();
        assert_eq!(
            Vec::from(&out),
            &[
                None,
                None,
                Some(3.0),
                Some(3.0),
                Some(2.0),
                Some(5.0),
                Some(11.0)
            ]
        );
    }

    #[test]
    fn test_rolling_var() {
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
        // window larger than array
        assert_eq!(ca.rolling_var(10).null_count(), ca.len());

        let out = ca.rolling_var(3).cast::<Int32Type>().unwrap();
        let out = out.i32().unwrap();
        assert_eq!(
            Vec::from(out),
            &[None, None, Some(1), None, None, None, None,]
        );

        let ca = Float64Chunked::new_from_slice("", &[0.0, 2.0, 8.0, 3.0, 12.0, 1.0]);
        let out = ca.rolling_var(3).cast::<Int32Type>().unwrap();
        let out = out.i32().unwrap();

        assert_eq!(
            Vec::from(out),
            &[None, None, Some(17), Some(10), Some(20), Some(34),]
        );
    }
}
