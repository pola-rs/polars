use num_traits::{FromPrimitive, ToPrimitive};

use super::no_nulls::RollingAggWindowNoNulls;
use super::nulls::RollingAggWindowNulls;
use super::*;
use crate::moment::{KurtosisState, SkewState, VarState};

pub trait RollingMomentAlgo {
    type State: Default + Clone;
    fn new(params: Option<RollingFnParams>) -> Self;
    fn insert_one(state: &mut Self::State, x: f64);
    fn combine(state: &mut Self::State, other: &Self::State);
    fn finalize(&self, state: &Self::State) -> Option<f64>;
}

pub struct VarianceMoment {
    ddof: u8,
}

impl RollingMomentAlgo for VarianceMoment {
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
    fn insert_one(state: &mut Self::State, x: f64) {
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
    fn insert_one(state: &mut Self::State, x: f64) {
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
    fn insert_one(state: &mut Self::State, x: f64) {
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

pub struct MomentWindow<'a, T, M: RollingMomentAlgo> {
    slice: &'a [T],
    validity: Option<&'a Bitmap>,
    moment: M,
    non_finite_count: usize, // NaN or infinity.
    null_count: usize,
    start: usize,
    end: usize,
    front: Vec<M::State>,
    back: Vec<f64>,
    agg_back: M::State,
}

impl<'a, T, M> MomentWindow<'a, T, M>
where
    T: NativeType + ToPrimitive + IsFloat + FromPrimitive,
    M: RollingMomentAlgo,
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
            non_finite_count: 0,
            null_count: 0,
            start: 0,
            end: 0,
            front: Vec::new(),
            back: Vec::new(),
            agg_back: M::State::default(),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.non_finite_count = 0;
        self.null_count = 0;
        self.front.clear();
        self.back.clear();
        self.agg_back = M::State::default();
    }

    #[inline(always)]
    fn push(&mut self, val: T) {
        if val.is_finite() {
            let x = NumCast::from(val).unwrap();
            self.back.push(x);
            M::insert_one(&mut self.agg_back, x);
        } else {
            // A hack to replicate ddof null behavior.
            self.back.push(0.0);
            M::insert_one(&mut self.agg_back, 0.0);
            self.non_finite_count += 1;
        }
    }

    #[inline(always)]
    fn pop(&mut self, val: T) {
        if self.front.is_empty() {
            self.flip();
        }
        self.front.pop();
        self.non_finite_count -= !val.is_finite() as usize;
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
    fn get_moment(&self) -> Option<T> {
        let mut state = self.agg_back.clone();
        if let Some(front_agg) = self.front.last() {
            M::combine(&mut state, front_agg);
        }
        let agg = self.moment.finalize(&state);
        if self.non_finite_count > 0 {
            agg.map(|_v| T::from_f64(f64::NAN).unwrap())
        } else {
            agg.map(|v| T::from_f64(v).unwrap())
        }
    }
}

impl<T, M> RollingAggWindowNoNulls<T> for MomentWindow<'_, T, M>
where
    T: NativeType + ToPrimitive + IsFloat + FromPrimitive,
    M: RollingMomentAlgo,
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
    M: RollingMomentAlgo,
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
