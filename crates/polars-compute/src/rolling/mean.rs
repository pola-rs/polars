use super::no_nulls::RollingAggWindowNoNulls;
use super::nulls::RollingAggWindowNulls;
use super::sum::SumWindow;
use super::*;

pub struct MeanWindow<'a, T> {
    sum: SumWindow<'a, T, f64>,
    current_agg: Option<T>,
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
            current_agg: None,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) {
        let sum = unsafe {
            RollingAggWindowNoNulls::update(&mut self.sum, start, end);
            RollingAggWindowNoNulls::get_agg(&self.sum, start).unwrap_unchecked()
        };
        self.current_agg = (start != end).then(|| sum / NumCast::from(end - start).unwrap());
    }

    fn get_agg(&self, _idx: usize) -> Option<T> {
        self.current_agg
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
            current_agg: None,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) {
        unsafe { RollingAggWindowNulls::update(&mut self.sum, start, end) };
        let sum = RollingAggWindowNulls::get_agg(&self.sum, start);
        let len = end - start;
        self.current_agg = if self.sum.null_count == len {
            None
        } else {
            sum.map(|sum| sum / NumCast::from(end - start - self.sum.null_count).unwrap())
        };
    }

    fn get_agg(&self, _idx: usize) -> Option<T> {
        self.current_agg
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        self.sum.is_valid(min_periods)
    }

    fn slice_len(&self) -> usize {
        RollingAggWindowNulls::slice_len(&self.sum)
    }
}
