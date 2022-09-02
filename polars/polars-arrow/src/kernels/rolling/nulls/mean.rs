use super::sum::SumWindow;
use super::{rolling_apply_agg_window, RollingAggWindowNulls, *};

pub struct MeanWindow<'a, T> {
    sum: SumWindow<'a, T>,
}

impl<
        'a,
        T: NativeType + IsFloat + Add<Output = T> + Sub<Output = T> + NumCast + Div<Output = T>,
    > RollingAggWindowNulls<'a, T> for MeanWindow<'a, T>
{
    unsafe fn new(slice: &'a [T], validity: &'a Bitmap, start: usize, end: usize) -> Self {
        Self {
            sum: SumWindow::new(slice, validity, start, end),
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        let sum = self.sum.update(start, end);
        sum.map(|sum| sum / NumCast::from(end - start - self.sum.null_count).unwrap())
    }
    fn is_valid(&self, min_periods: usize) -> bool {
        self.sum.is_valid(min_periods)
    }
}

pub fn rolling_mean<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType
        + IsFloat
        + PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + NumCast
        + Div<Output = T>,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply_agg_window::<MeanWindow<_>, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
        )
    } else {
        rolling_apply_agg_window::<MeanWindow<_>, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
        )
    }
}
