use super::*;

pub(super) struct SumSquaredWindow<'a, T> {
    slice: &'a [T],
    validity: &'a Bitmap,
    sum_of_squares: Option<T>,
    last_start: usize,
    last_end: usize,
    null_count: usize,
}

impl<'a, T: NativeType + IsFloat + Add<Output = T> + Sub<Output = T> + Mul<Output = T>>
    SumSquaredWindow<'a, T>
{
    // compute sum from the entire window
    unsafe fn compute_sum_and_null_count(&mut self, start: usize, end: usize) -> Option<T> {
        let mut sum_of_squares = None;
        let mut idx = start;
        self.null_count = 0;
        for value in &self.slice[start..end] {
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
    RollingAggWindowNulls<'a, T> for SumSquaredWindow<'a, T>
{
    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        _params: DynArgs,
    ) -> Self {
        let mut out = Self {
            slice,
            validity,
            sum_of_squares: None,
            last_start: start,
            last_end: end,
            null_count: 0,
        };
        out.compute_sum_and_null_count(start, end);
        out
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        let recompute_sum = if start >= self.last_end {
            true
        } else {
            // remove elements that should leave the window
            let mut recompute_sum = false;
            for idx in self.last_start..start {
                // SAFETY:
                // we are in bounds
                let valid = self.validity.get_bit_unchecked(idx);
                if valid {
                    let leaving_value = *self.slice.get_unchecked(idx);

                    // if the leaving value is nan we need to recompute the window
                    if T::is_float() && !leaving_value.is_finite() {
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
            recompute_sum
        };

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
        self.sum_of_squares
    }
    fn is_valid(&self, min_periods: usize) -> bool {
        ((self.last_end - self.last_start) - self.null_count) >= min_periods
    }
}

// E[(xi - E[x])^2]
// can be expanded to
// E[x^2] - E[x]^2
pub struct VarWindow<'a, T> {
    mean: MeanWindow<'a, T>,
    sum_of_squares: SumSquaredWindow<'a, T>,
    ddof: u8,
}

impl<
        'a,
        T: NativeType
            + IsFloat
            + Float
            + std::iter::Sum
            + AddAssign
            + SubAssign
            + Div<Output = T>
            + NumCast
            + One
            + Zero
            + PartialOrd
            + Add<Output = T>
            + Sub<Output = T>,
    > RollingAggWindowNulls<'a, T> for VarWindow<'a, T>
{
    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        params: DynArgs,
    ) -> Self {
        Self {
            mean: MeanWindow::new(slice, validity, start, end, None),
            sum_of_squares: SumSquaredWindow::new(slice, validity, start, end, None),
            ddof: match params {
                None => 1,
                Some(pars) => pars.downcast_ref::<RollingVarParams>().unwrap().ddof,
            },
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        let sum_of_squares = self.sum_of_squares.update(start, end)?;
        let null_count = self.sum_of_squares.null_count;
        let count: T = NumCast::from(end - start - null_count).unwrap();

        let mean = self.mean.update(start, end)?;
        let ddof = NumCast::from(self.ddof).unwrap();

        let denom = count - ddof;

        if count == T::zero() {
            None
        } else if count == T::one() {
            NumCast::from(0)
        } else if denom <= T::zero() {
            Some(T::infinity())
        } else {
            let var = (sum_of_squares - count * mean * mean) / denom;
            Some(if var < T::zero() { T::zero() } else { var })
        }
    }
    fn is_valid(&self, min_periods: usize) -> bool {
        self.mean.is_valid(min_periods)
    }
}

pub fn rolling_var<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    params: DynArgs,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum<T> + Zero + AddAssign + SubAssign + IsFloat + Float,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    let offsets_fn = if center {
        det_offsets_center
    } else {
        det_offsets
    };
    rolling_apply_agg_window::<VarWindow<_>, _, _>(
        arr.values().as_slice(),
        arr.validity().as_ref().unwrap(),
        window_size,
        min_periods,
        offsets_fn,
        params,
    )
}
