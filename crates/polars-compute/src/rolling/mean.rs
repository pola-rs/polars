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

    fn is_valid(&self, min_periods: usize) -> bool {
        self.sum.is_valid(min_periods)
    }

    fn slice_len(&self) -> usize {
        RollingAggWindowNulls::slice_len(&self.sum)
    }
}
