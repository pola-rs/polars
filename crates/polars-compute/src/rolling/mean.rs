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

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        let sum = unsafe {
            RollingAggWindowNoNulls::update(&mut self.sum, start, end).unwrap_unchecked()
        };
        if start == end {
            None
        } else {
            Some(sum / NumCast::from(end - start).unwrap())
        }
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

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        let sum = unsafe { RollingAggWindowNulls::update(&mut self.sum, start, end) };
        let len = end - start;
        if self.sum.null_count == len {
            None
        } else {
            sum.map(|sum| sum / NumCast::from(end - start - self.sum.null_count).unwrap())
        }
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        self.sum.is_valid(min_periods)
    }

    fn slice_len(&self) -> usize {
        RollingAggWindowNulls::slice_len(&self.sum)
    }
}
