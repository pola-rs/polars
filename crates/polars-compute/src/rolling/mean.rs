use std::cell::Cell;

use super::no_nulls::RollingAggWindowNoNulls;
use super::nulls::RollingAggWindowNulls;
use super::sum::SumWindow;
use super::*;

pub struct MeanWindow<'a, T> {
    sum: SumWindow<'a, T, f64>,
}

impl<T> RollingAggWindowNoNulls<T> for MeanWindow<'_, T>
where
    T: NativeType
        + IsFloat
        + std::iter::Sum
        + AddAssign
        + SubAssign
        + Div<Output = T>
        + NumCast
        + Add<Output = T>
        + Sub<Output = T>
        + PartialOrd,
{
    type This<'a> = MeanWindow<'a, T>;

    fn new<'a>(
        slice: &'a [T],
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        window_size: Option<usize>,
    ) -> Self::This<'a> {
        MeanWindow {
            sum: <SumWindow<T, f64> as RollingAggWindowNoNulls<T>>::new(
                slice,
                start,
                end,
                params,
                window_size,
            ),
        }
    }

    unsafe fn update(&mut self, new_start: usize, new_end: usize) {
        unsafe {
            RollingAggWindowNoNulls::update(&mut self.sum, new_start, new_end);
        };
    }

    fn get_agg(&self, idx: usize) -> Option<T> {
        let sum = RollingAggWindowNoNulls::get_agg(&self.sum, idx).unwrap();
        (self.sum.start != self.sum.end)
            .then(|| sum / NumCast::from(self.sum.end - self.sum.start).unwrap())
    }

    fn slice_len(&self) -> usize {
        RollingAggWindowNulls::slice_len(&self.sum)
    }
}

impl<
    T: NativeType
        + IsFloat
        + Add<Output = T>
        + Sub<Output = T>
        + NumCast
        + Div<Output = T>
        + AddAssign
        + SubAssign
        + PartialOrd,
> RollingAggWindowNulls<T> for MeanWindow<'_, T>
{
    type This<'a> = MeanWindow<'a, T>;

    fn new<'a>(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        window_size: Option<usize>,
    ) -> Self::This<'a> {
        MeanWindow {
            sum: <SumWindow<T, f64> as RollingAggWindowNulls<T>>::new(
                slice,
                validity,
                start,
                end,
                params,
                window_size,
            ),
        }
    }

    unsafe fn update(&mut self, new_start: usize, new_end: usize) {
        unsafe { RollingAggWindowNulls::update(&mut self.sum, new_start, new_end) };
    }

    fn get_agg(&self, idx: usize) -> Option<T> {
        let sum = RollingAggWindowNulls::get_agg(&self.sum, idx);
        let len = self.sum.end - self.sum.start;
        if self.sum.null_count == len {
            None
        } else {
            sum.map(|sum| {
                sum / NumCast::from(self.sum.end - self.sum.start - self.sum.null_count).unwrap()
            })
        }
    }

    fn is_valid(&self, min_samples: usize) -> bool {
        self.sum.is_valid(min_samples)
    }

    fn slice_len(&self) -> usize {
        RollingAggWindowNulls::slice_len(&self.sum)
    }
}

/// An exact mean of temporal values stored as `i64` ticks.
///
/// The mean is rounded towards negative infinity if `FLOOR` is set and towards zero otherwise.
/// It is multiplied by `SCALE` before it is rounded, which is how `Date` days become `Datetime(us)`
/// (so `FLOOR` has to be set if `SCALE` is not 1).
pub struct ExactMeanWindow<'a, const FLOOR: bool, const SCALE: i64> {
    sum: SumWindow<'a, i64, i128>,
    /// Any value gives the right mean, but it is fastest if it is close to the mean of the window:
    /// then `sum - count * base` fits an `i64` and the division below is not a 128-bit one.
    base: Cell<i64>,
}

impl<const FLOOR: bool, const SCALE: i64> ExactMeanWindow<'_, FLOOR, SCALE> {
    /// The mean of the `count` valid values in the window.
    fn mean(&self, count: usize) -> Option<i64> {
        const { assert!(FLOOR || SCALE == 1) };
        if count == 0 {
            return None;
        }
        let sum = self.sum.sum;
        let count = count as i64;
        let mut base = self.base.get();
        let offset = match i64::try_from(sum - count as i128 * base as i128) {
            Ok(offset) => offset,
            Err(_) => {
                // The floor of the mean of `i64` values is an `i64`.
                base = sum.div_euclid(count as i128) as i64;
                self.base.set(base);
                (sum - count as i128 * base as i128) as i64
            },
        };

        // `sum = count * base + offset`, so `base + quotient` is `floor(sum / count)` and
        // `remainder` is `sum mod count`.
        let mut quotient = offset / count;
        let mut remainder = offset % count;
        if remainder < 0 {
            quotient -= 1;
            remainder += count;
        }
        let floor = base + quotient;
        if SCALE == 1 {
            // Rounding towards zero only differs for inexact negative means.
            return Some(if FLOOR || sum >= 0 || remainder == 0 {
                floor
            } else {
                floor + 1
            });
        }
        // `floor(sum * SCALE / count) = floor * SCALE + floor(remainder * SCALE / count)`.
        let fraction = match remainder.checked_mul(SCALE) {
            Some(scaled) => scaled / count,
            None => (remainder as i128 * SCALE as i128 / count as i128) as i64,
        };
        i64::try_from(floor as i128 * SCALE as i128 + fraction as i128).ok()
    }
}

impl<const FLOOR: bool, const SCALE: i64> RollingAggWindowNoNulls<i64, i64>
    for ExactMeanWindow<'_, FLOOR, SCALE>
{
    type This<'a> = ExactMeanWindow<'a, FLOOR, SCALE>;

    fn new<'a>(
        slice: &'a [i64],
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        window_size: Option<usize>,
    ) -> Self::This<'a> {
        ExactMeanWindow {
            sum: <SumWindow<i64, i128> as RollingAggWindowNoNulls<i64>>::new(
                slice,
                start,
                end,
                params,
                window_size,
            ),
            base: Cell::new(0),
        }
    }

    unsafe fn update(&mut self, new_start: usize, new_end: usize) {
        unsafe { RollingAggWindowNoNulls::update(&mut self.sum, new_start, new_end) };
    }

    fn get_agg(&self, _idx: usize) -> Option<i64> {
        self.mean(self.sum.end - self.sum.start)
    }

    fn slice_len(&self) -> usize {
        RollingAggWindowNoNulls::slice_len(&self.sum)
    }
}

impl<const FLOOR: bool, const SCALE: i64> RollingAggWindowNulls<i64, i64>
    for ExactMeanWindow<'_, FLOOR, SCALE>
{
    type This<'a> = ExactMeanWindow<'a, FLOOR, SCALE>;

    fn new<'a>(
        slice: &'a [i64],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        window_size: Option<usize>,
    ) -> Self::This<'a> {
        ExactMeanWindow {
            sum: <SumWindow<i64, i128> as RollingAggWindowNulls<i64>>::new(
                slice,
                validity,
                start,
                end,
                params,
                window_size,
            ),
            base: Cell::new(0),
        }
    }

    unsafe fn update(&mut self, new_start: usize, new_end: usize) {
        unsafe { RollingAggWindowNulls::update(&mut self.sum, new_start, new_end) };
    }

    fn get_agg(&self, _idx: usize) -> Option<i64> {
        self.mean(self.sum.end - self.sum.start - self.sum.null_count)
    }

    fn slice_len(&self) -> usize {
        RollingAggWindowNulls::slice_len(&self.sum)
    }

    fn is_valid(&self, min_samples: usize) -> bool {
        self.sum.is_valid(min_samples)
    }
}
