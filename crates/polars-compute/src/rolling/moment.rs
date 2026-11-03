use num_traits::{FromPrimitive, ToPrimitive};

use super::no_nulls::RollingAggWindowNoNulls;
use super::nulls::RollingAggWindowNulls;
use super::*;
use crate::moment::{CovState, KurtosisState, PearsonState, SkewState, VarState};

pub trait RollingMomentAlgo {
    type Input: Copy + Default;
    type State: Default + Clone;
    fn new(params: Option<RollingFnParams>) -> Self;
    fn insert_one(state: &mut Self::State, x: Self::Input);
    fn combine(state: &mut Self::State, other: &Self::State);
    fn finalize(&self, state: &Self::State) -> Option<f64>;
}

pub struct VarianceMoment {
    ddof: u8,
}

impl RollingMomentAlgo for VarianceMoment {
    type Input = f64;
    type State = VarState;

    fn new(params: Option<RollingFnParams>) -> Self {
        let ddof = if let Some(RollingFnParams::Var(params)) = params {
            params.ddof
        } else {
            1
        };

        Self { ddof }
    }

    #[inline(always)]
    fn insert_one(state: &mut Self::State, x: Self::Input) {
        state.insert_one(x);
    }

    #[inline(always)]
    fn combine(state: &mut Self::State, other: &Self::State) {
        state.combine(other);
    }

    #[inline(always)]
    fn finalize(&self, state: &Self::State) -> Option<f64> {
        state.finalize(self.ddof)
    }
}

pub struct KurtosisMoment {
    fisher: bool,
    bias: bool,
}

impl RollingMomentAlgo for KurtosisMoment {
    type Input = f64;
    type State = KurtosisState;

    fn new(params: Option<RollingFnParams>) -> Self {
        let (fisher, bias) = if let Some(RollingFnParams::Kurtosis { fisher, bias }) = params {
            (fisher, bias)
        } else {
            (false, false)
        };

        Self { fisher, bias }
    }

    #[inline(always)]
    fn insert_one(state: &mut Self::State, x: Self::Input) {
        state.insert_one(x);
    }

    #[inline(always)]
    fn combine(state: &mut Self::State, other: &Self::State) {
        state.combine(other);
    }

    #[inline(always)]
    fn finalize(&self, state: &Self::State) -> Option<f64> {
        state.finalize(self.fisher, self.bias)
    }
}

pub struct SkewMoment {
    bias: bool,
}

impl RollingMomentAlgo for SkewMoment {
    type Input = f64;
    type State = SkewState;

    fn new(params: Option<RollingFnParams>) -> Self {
        let bias = if let Some(RollingFnParams::Skew { bias }) = params {
            bias
        } else {
            false
        };

        Self { bias }
    }

    #[inline(always)]
    fn insert_one(state: &mut Self::State, x: Self::Input) {
        state.insert_one(x);
    }

    #[inline(always)]
    fn combine(state: &mut Self::State, other: &Self::State) {
        state.combine(other);
    }

    #[inline(always)]
    fn finalize(&self, state: &Self::State) -> Option<f64> {
        state.finalize(self.bias)
    }
}

/// Aggregates a window with two stacks, so values leave the window without being subtracted.
struct TwoStack<M: RollingMomentAlgo> {
    non_finite_count: usize, // NaN or infinity.
    front: Vec<M::State>,
    back: Vec<M::Input>,
    agg_back: M::State,
}

impl<M: RollingMomentAlgo> TwoStack<M> {
    fn new() -> Self {
        Self {
            non_finite_count: 0,
            front: Vec::new(),
            back: Vec::new(),
            agg_back: M::State::default(),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.non_finite_count = 0;
        self.front.clear();
        self.back.clear();
        self.agg_back = M::State::default();
    }

    #[inline(always)]
    fn push(&mut self, x: M::Input, is_finite: bool) {
        let x = if is_finite {
            x
        } else {
            // A hack to replicate ddof null behavior.
            self.non_finite_count += 1;
            M::Input::default()
        };
        self.back.push(x);
        M::insert_one(&mut self.agg_back, x);
    }

    #[inline(always)]
    fn pop(&mut self, is_finite: bool) {
        if self.front.is_empty() {
            self.flip();
        }
        self.front.pop();
        self.non_finite_count -= !is_finite as usize;
    }

    fn flip(&mut self) {
        let mut agg = M::State::default();
        while let Some(x) = self.back.pop() {
            M::insert_one(&mut agg, x);
            self.front.push(agg.clone());
        }

        self.agg_back = M::State::default();
    }

    #[inline(always)]
    fn get_moment<T: FromPrimitive>(&self, moment: &M) -> Option<T> {
        let mut state = self.agg_back.clone();
        if let Some(front_agg) = self.front.last() {
            M::combine(&mut state, front_agg);
        }
        let agg = moment.finalize(&state)?;
        let agg = if self.non_finite_count > 0 {
            f64::NAN
        } else {
            agg
        };
        Some(T::from_f64(agg).unwrap())
    }
}

pub struct MomentWindow<'a, T, M: RollingMomentAlgo> {
    slice: &'a [T],
    validity: Option<&'a Bitmap>,
    moment: M,
    null_count: usize,
    start: usize,
    end: usize,
    stack: TwoStack<M>,
}

impl<'a, T, M> MomentWindow<'a, T, M>
where
    T: NativeType + ToPrimitive + IsFloat + FromPrimitive,
    M: RollingMomentAlgo<Input = f64>,
{
    fn new_impl(
        slice: &'a [T],
        validity: Option<&'a Bitmap>,
        params: Option<RollingFnParams>,
    ) -> Self {
        Self {
            slice,
            validity,
            moment: M::new(params),
            null_count: 0,
            start: 0,
            end: 0,
            stack: TwoStack::new(),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.null_count = 0;
        self.stack.reset();
    }

    #[inline(always)]
    fn push(&mut self, val: T) {
        self.stack
            .push(NumCast::from(val).unwrap(), val.is_finite());
    }

    #[inline(always)]
    fn pop(&mut self, val: T) {
        self.stack.pop(val.is_finite());
    }

    #[inline(always)]
    fn get_moment(&self) -> Option<T> {
        self.stack.get_moment(&self.moment)
    }
}

impl<T, M> RollingAggWindowNoNulls<T> for MomentWindow<'_, T, M>
where
    T: NativeType + ToPrimitive + IsFloat + FromPrimitive,
    M: RollingMomentAlgo<Input = f64>,
{
    type This<'a> = MomentWindow<'a, T, M>;

    fn new<'a>(
        slice: &'a [T],
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        _window_size: Option<usize>,
    ) -> Self::This<'a> {
        let mut out = MomentWindow::new_impl(slice, None, params);
        unsafe { RollingAggWindowNoNulls::update(&mut out, start, end) };
        out
    }

    // # Safety
    // The start, end range must be in-bounds.
    #[inline]
    unsafe fn update(&mut self, new_start: usize, new_end: usize) {
        if new_start >= self.end {
            self.reset();
            self.start = new_start;
            self.end = new_start;
        }

        for val in &self.slice[self.start..new_start] {
            self.pop(*val);
        }

        for val in &self.slice[self.end..new_end] {
            self.push(*val);
        }

        self.start = new_start;
        self.end = new_end;
    }

    fn get_agg(&self, _idx: usize) -> Option<T> {
        self.get_moment()
    }

    fn slice_len(&self) -> usize {
        self.slice.len()
    }
}

impl<T, M> RollingAggWindowNulls<T> for MomentWindow<'_, T, M>
where
    T: NativeType + ToPrimitive + IsFloat + FromPrimitive,
    M: RollingMomentAlgo<Input = f64>,
{
    type This<'a> = MomentWindow<'a, T, M>;

    fn new<'a>(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        _window_size: Option<usize>,
    ) -> Self::This<'a> {
        assert!(start <= slice.len() && end <= slice.len() && start <= end);
        let mut out = MomentWindow::new_impl(slice, Some(validity), params);
        // SAFETY: We bounds checked `start` and `end`.
        unsafe { RollingAggWindowNulls::update(&mut out, start, end) };
        out
    }

    // # Safety
    // The start, end range must be in-bounds.
    #[inline]
    unsafe fn update(&mut self, new_start: usize, new_end: usize) {
        let validity = unsafe { self.validity.unwrap_unchecked() };

        if new_start >= self.end {
            self.reset();
            self.start = new_start;
            self.end = new_start;
        }

        for idx in self.start..new_start {
            let valid = unsafe { validity.get_bit_unchecked(idx) };
            if valid {
                self.pop(unsafe { *self.slice.get_unchecked(idx) });
            } else {
                self.null_count -= 1;
            }
        }

        for idx in self.end..new_end {
            let valid = unsafe { validity.get_bit_unchecked(idx) };
            if valid {
                self.push(unsafe { *self.slice.get_unchecked(idx) });
            } else {
                self.null_count += 1;
            }
        }

        self.start = new_start;
        self.end = new_end;
    }

    fn get_agg(&self, _idx: usize) -> Option<T> {
        self.get_moment()
    }

    #[inline(always)]
    fn is_valid(&self, min_periods: usize) -> bool {
        ((self.end - self.start) - self.null_count) >= min_periods
    }

    fn slice_len(&self) -> usize {
        self.slice.len()
    }
}

struct CovarianceMoment {
    ddof: u8,
}

impl RollingMomentAlgo for CovarianceMoment {
    type Input = (f64, f64);
    type State = CovState;

    fn new(params: Option<RollingFnParams>) -> Self {
        let ddof = if let Some(RollingFnParams::Var(params)) = params {
            params.ddof
        } else {
            1
        };

        Self { ddof }
    }

    #[inline(always)]
    fn insert_one(state: &mut Self::State, (x, y): Self::Input) {
        state.insert_one(x, y);
    }

    #[inline(always)]
    fn combine(state: &mut Self::State, other: &Self::State) {
        state.combine(other);
    }

    #[inline(always)]
    fn finalize(&self, state: &Self::State) -> Option<f64> {
        // A non-empty window with at most `ddof` values gives NaN.
        (state.weight() > 0.0).then(|| state.finalize(self.ddof).unwrap_or(f64::NAN))
    }
}

struct PearsonMoment;

impl RollingMomentAlgo for PearsonMoment {
    type Input = (f64, f64);
    type State = PearsonState;

    fn new(_params: Option<RollingFnParams>) -> Self {
        Self
    }

    #[inline(always)]
    fn insert_one(state: &mut Self::State, (x, y): Self::Input) {
        state.insert_one(x, y);
    }

    #[inline(always)]
    fn combine(state: &mut Self::State, other: &Self::State) {
        state.combine(other);
    }

    #[inline(always)]
    fn finalize(&self, state: &Self::State) -> Option<f64> {
        (state.weight() > 1.0).then(|| state.finalize())
    }
}

/// Rolling moment over the pairs of `x` and `y`, ignoring the pairs that are unset in `validity`.
fn rolling_co_moment<T, M>(
    x: &[T],
    y: &[T],
    validity: Option<&Bitmap>,
    window_size: usize,
    min_periods: usize,
    params: Option<RollingFnParams>,
) -> ArrayRef
where
    T: NativeType + ToPrimitive + IsFloat + FromPrimitive,
    M: RollingMomentAlgo<Input = (f64, f64)>,
{
    assert_eq!(x.len(), y.len());
    assert!(validity.is_none_or(|v| v.len() == x.len()));
    let len = x.len();
    let moment = M::new(params);
    let mut stack = TwoStack::<M>::new();
    let mut null_count = 0;
    let mut start = 0;
    let mut end = 0;
    let out = (0..len).map(|idx| {
        let (new_start, new_end) = det_offsets(idx, window_size, len);
        if new_start >= end {
            stack.reset();
            null_count = 0;
            start = new_start;
            end = new_start;
        }
        let is_valid = |i: usize| validity.is_none_or(|v| v.get_bit(i));
        for i in start..new_start {
            if is_valid(i) {
                stack.pop(x[i].is_finite() && y[i].is_finite());
            } else {
                null_count -= 1;
            }
        }
        for i in end..new_end {
            if is_valid(i) {
                let pair = (x[i].to_f64().unwrap(), y[i].to_f64().unwrap());
                stack.push(pair, x[i].is_finite() && y[i].is_finite());
            } else {
                null_count += 1;
            }
        }
        (start, end) = (new_start, new_end);

        if (end - start) - null_count < min_periods {
            return None;
        }
        stack.get_moment::<T>(&moment)
    });
    Box::new(PrimitiveArray::from_trusted_len_iter(out))
}

/// Rolling covariance of `x` and `y`, ignoring the pairs that are unset in `validity`.
pub fn rolling_cov<T>(
    x: &[T],
    y: &[T],
    validity: Option<&Bitmap>,
    window_size: usize,
    min_periods: usize,
    ddof: u8,
) -> ArrayRef
where
    T: NativeType + ToPrimitive + IsFloat + FromPrimitive,
{
    let params = Some(RollingFnParams::Var(RollingVarParams { ddof }));
    rolling_co_moment::<T, CovarianceMoment>(x, y, validity, window_size, min_periods, params)
}

/// Rolling Pearson correlation of `x` and `y`, ignoring the pairs that are unset in `validity`.
pub fn rolling_corr<T>(
    x: &[T],
    y: &[T],
    validity: Option<&Bitmap>,
    window_size: usize,
    min_periods: usize,
) -> ArrayRef
where
    T: NativeType + ToPrimitive + IsFloat + FromPrimitive,
{
    rolling_co_moment::<T, PearsonMoment>(x, y, validity, window_size, min_periods, None)
}
