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

/// An exact mean of temporal values stored as i64 ticks.
pub struct ExactMeanWindow<'a, const FLOOR: bool, const SCALE: i128> {
    sum: SumWindow<'a, i64, i128>,
}

impl<const FLOOR: bool, const SCALE: i128> ExactMeanWindow<'_, FLOOR, SCALE> {
    fn mean(&self, count: usize) -> Option<i64> {
        if count == 0 {
            return None;
        }
        let count = count as i128;
        let quotient = if FLOOR {
            self.sum.sum.div_euclid(count)
        } else {
            self.sum.sum / count
        };
        let remainder = self.sum.sum - quotient * count;
        let mean = quotient * SCALE + (remainder * SCALE) / count;
        i64::try_from(mean).ok()
    }
}

impl<const FLOOR: bool, const SCALE: i128> RollingAggWindowNoNulls<i64, i64>
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

impl<const FLOOR: bool, const SCALE: i128> RollingAggWindowNulls<i64, i64>
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
