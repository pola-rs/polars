use core::panic;
use std::marker::PhantomData;

use polars_utils::IdxSize;
use polars_utils::order_statistic_tree::OrderStatisticTree;

use super::super::rank::*;
use super::*;

pub struct RankWindow<'a, T, Out, P> {
    slice: &'a [T],
    validity: &'a Bitmap,
    start: usize,
    end: usize,
    ost: OrderStatisticTree<&'a T>,
    policy: P,
    _out: PhantomData<Out>,
}

impl<T, Out, P> RollingAggWindowNulls<T, Out> for RankWindow<'_, T, Out, P>
where
    T: NativeType,
    Out: NativeType,
    P: RankPolicy<T, Out>,
{
    type This<'a> = RankWindow<'a, T, Out, P>;

    fn new<'a>(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        window_size: Option<usize>,
    ) -> Self::This<'a> {
        assert!(start <= slice.len() && end <= slice.len() && start <= end);

        let cmp = |a: &&T, b: &&T| T::tot_cmp(*a, *b);
        let ost: OrderStatisticTree<&T> = match window_size {
            Some(ws) => OrderStatisticTree::with_capacity(ws, cmp),
            None => OrderStatisticTree::new(cmp),
        };
        let mut this = RankWindow {
            slice,
            validity,
            start: 0,
            end: 0,
            ost,
            policy: P::new(&params.unwrap()),
            _out: PhantomData,
        };
        // SAFETY: We bounds checked `start` and `end`.
        unsafe {
            this.update(start, end);
        }
        this
    }

    unsafe fn update(&mut self, new_start: usize, new_end: usize) {
        debug_assert!(self.start <= self.end);
        debug_assert!(self.end <= self.slice.len());
        debug_assert!(new_start <= new_end);
        debug_assert!(new_end <= self.slice.len());
        debug_assert!(self.start <= new_start);
        debug_assert!(self.end <= new_end);

        for i in self.end..new_end {
            if !self.validity.get(i).unwrap() {
                continue;
            }
            self.ost.insert(unsafe { self.slice.get_unchecked(i) });
        }
        for i in self.start..new_start {
            if !self.validity.get(i).unwrap() {
                continue;
            }
            self.ost
                .remove(&unsafe { self.slice.get_unchecked(i) })
                .expect("previously added value is missing");
        }
        self.start = new_start;
        self.end = new_end;
    }

    fn get_agg(&self, idx: usize) -> Option<Out> {
        if !(self.start..self.end).contains(&idx) {
            panic!("index out of bounds");
        }
        self.policy.rank(&self.ost, &self.slice[idx])
    }

    fn is_valid(&self, _min_periods: usize) -> bool {
        self.validity.get(self.end - 1).unwrap()
    }

    fn slice_len(&self) -> usize {
        self.slice.len()
    }
}

pub type RankWindowAvg<'a, T> = RankWindow<'a, T, f64, RankPolicyAverage>;
pub type RankWindowMin<'a, T> = RankWindow<'a, T, IdxSize, RankPolicyMin>;
pub type RankWindowMax<'a, T> = RankWindow<'a, T, IdxSize, RankPolicyMax>;
pub type RankWindowDense<'a, T> = RankWindow<'a, T, IdxSize, RankPolicyDense>;
pub type RankWindowRandom<'a, T> = RankWindow<'a, T, IdxSize, RankPolicyRandom>;

pub fn rolling_rank<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    params: Option<RollingFnParams>,
) -> ArrayRef
where
    T: NativeType,
{
    assert!(weights.is_none(), "weights are not supported for rank");

    let offset_fn = match center {
        true => det_offsets_center,
        false => det_offsets,
    };
    let method = if let Some(RollingFnParams::Rank { method, .. }) = params {
        method
    } else {
        unreachable!("expected RollingFnParams::Rank");
    };

    match method {
        RollingRankMethod::Average => rolling_apply_agg_window::<RankWindowAvg<T>, _, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            offset_fn,
            params,
        ),
        RollingRankMethod::Min => rolling_apply_agg_window::<RankWindowMin<T>, _, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            offset_fn,
            params,
        ),
        RollingRankMethod::Max => rolling_apply_agg_window::<RankWindowMax<T>, _, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            offset_fn,
            params,
        ),
        RollingRankMethod::Dense => rolling_apply_agg_window::<RankWindowDense<T>, _, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            offset_fn,
            params,
        ),
        RollingRankMethod::Random => rolling_apply_agg_window::<RankWindowRandom<T>, _, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            offset_fn,
            params,
        ),
    }
}
