use super::*;
use mean::MeanWindow;
use nulls;
use nulls::{rolling_apply_agg_window, RollingAggWindow};

pub struct SumSquaredWindow<'a, T> {
    slice: &'a [T],
    validity: &'a Bitmap,
    sum_of_squares: Option<T>,
    last_start: usize,
    last_end: usize,
    null_count: usize,
    min_periods: usize,
}

impl<'a, T: NativeType + IsFloat + Add<Output = T> + Sub<Output = T> + Mul<Output = T>>
    SumSquaredWindow<'a, T>
{
    // compute sum from the entire window
    unsafe fn compute_sum_and_null_count(&mut self, start: usize, end: usize) -> Option<T> {
        let mut sum_of_squares = None;
        let mut idx = start;
        self.null_count = 0;
        for value in (&self.slice[start..end]).iter() {
            let valid = self.validity.get_bit_unchecked(idx);
            if valid {
                match sum_of_squares {
                    None => sum_of_squares = Some(*value * *value),
                    Some(current) => sum_of_squares = Some(*value * *value + current),
                }
            } else {
                self.null_count += 1;
            }
            idx += 1;
        }
        self.sum_of_squares = sum_of_squares;
        sum_of_squares
    }
}

impl<'a, T: NativeType + IsFloat + Add<Output = T> + Sub<Output = T> + Mul<Output = T>>
    RollingAggWindow<'a, T> for SumSquaredWindow<'a, T>
{
    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        min_periods: usize,
    ) -> Self {
        let mut out = Self {
            slice,
            validity,
            sum_of_squares: None,
            last_start: start,
            last_end: end,
            null_count: 0,
            min_periods,
        };
        out.compute_sum_and_null_count(start, end);
        out
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        // remove elements that should leave the window
        let mut recompute_sum = false;
        for idx in self.last_start..start {
            // safety
            // we are in bounds
            let valid = self.validity.get_bit_unchecked(idx);
            if valid {
                let leaving_value = *self.slice.get_unchecked(idx);

                // if the leaving value is nan we need to recompute the window
                if T::is_float() && leaving_value.is_nan() {
                    recompute_sum = true;
                    break;
                }
                self.sum_of_squares = self
                    .sum_of_squares
                    .map(|v| v - leaving_value * leaving_value)
            } else {
                // null value leaving the window
                self.null_count -= 1;

                // self.sum is None and the leaving value is None
                // if the entering value is valid, we might get a new sum.
                if self.sum_of_squares.is_none() {
                    recompute_sum = true;
                    break;
                }
            }
        }
        self.last_start = start;

        // we traverse all values and compute
        if recompute_sum {
            self.compute_sum_and_null_count(start, end);
        } else {
            for idx in self.last_end..end {
                let valid = self.validity.get_bit_unchecked(idx);

                if valid {
                    let value = *self.slice.get_unchecked(idx);
                    let value = value * value;
                    match self.sum_of_squares {
                        None => self.sum_of_squares = Some(value),
                        Some(current) => self.sum_of_squares = Some(current + value),
                    }
                } else {
                    // null value entering the window
                    self.null_count += 1;
                }
            }
        }
        self.last_end = end;
        if ((end - start) - self.null_count) < self.min_periods {
            None
        } else {
            self.sum_of_squares
        }
    }
}

// E[(xi - E[x])^2]
// can be expanded to
// E[x^2] - E[x]^2
struct VarWindow<'a, T> {
    mean: MeanWindow<'a, T>,
    sum_of_squares: SumSquaredWindow<'a, T>,
}

impl<
        'a,
        T: NativeType
            + IsFloat
            + std::iter::Sum
            + AddAssign
            + SubAssign
            + Div<Output = T>
            + NumCast
            + One
            + Add<Output = T>
            + Sub<Output = T>,
    > RollingAggWindow<'a, T> for VarWindow<'a, T>
{
    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        min_periods: usize,
    ) -> Self {
        Self {
            mean: MeanWindow::new(slice, validity, start, end, min_periods),
            sum_of_squares: SumSquaredWindow::new(slice, validity, start, end, min_periods),
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        let sum_of_squares = self.sum_of_squares.update(start, end)?;
        let null_count = self.sum_of_squares.null_count;
        let count = NumCast::from(end - start - null_count).unwrap();

        let mean_of_squares = sum_of_squares / count;
        let mean = self.mean.update(start, end)?;
        let var = mean_of_squares - mean * mean;
        // apply Bessel's correction
        Some(var / (count - T::one()) * count)
    }
}

pub fn rolling_var<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum<T> + Zero + AddAssign + SubAssign + IsFloat + Float,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply_agg_window::<VarWindow<_>, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
        )
    } else {
        rolling_apply_agg_window::<VarWindow<_>, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
        )
    }
}
