use std::marker::PhantomData;

use polars_utils::IdxSize;
use polars_utils::order_statistic_tree::OrderStatisticTree;

use super::super::rank::*;
use super::*;

#[derive(Debug)]
pub struct RankWindow<'a, T, Out, P>
where
    T: NativeType,
    Out: NativeType,
    P: RankPolicy<T, Out>,
{
    slice: &'a [T],
    last_start: usize,
    last_end: usize,
    ost: OrderStatisticTree<&'a T>,
    policy: P,
    _out: PhantomData<Out>,
}

impl<'a, T, Out, P> RollingAggWindowNoNulls<'a, T, Out> for RankWindow<'a, T, Out, P>
where
    T: NativeType,
    Out: NativeType,
    P: RankPolicy<T, Out>,
{
    fn new(
        slice: &'a [T],
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        window_size: Option<usize>,
    ) -> Self {
        let cmp = |a: &&T, b: &&T| T::tot_cmp(*a, *b);
        let ost = match window_size {
            Some(ws) => OrderStatisticTree::with_capacity(ws, cmp),
            None => OrderStatisticTree::new(cmp),
        };
        let policy = P::new(&params.unwrap());
        let mut slf = Self {
            slice,
            last_start: 0,
            last_end: 0,
            ost,
            policy,
            _out: PhantomData,
        };
        unsafe {
            slf.update(start, end);
        }
        slf
    }

    unsafe fn update(&mut self, new_start: usize, new_end: usize) -> Option<Out> {
        debug_assert!(self.ost.len() == self.last_end - self.last_start);
        debug_assert!(self.last_start <= self.last_end);
        debug_assert!(self.last_end <= self.slice.len());
        debug_assert!(new_start <= new_end);
        debug_assert!(new_end <= self.slice.len());
        debug_assert!(self.last_start <= new_start);
        debug_assert!(self.last_end <= new_end);

        for i in self.last_end..new_end {
            self.ost.insert(unsafe { self.slice.get_unchecked(i) });
        }
        for i in self.last_start..new_start {
            self.ost
                .remove(&unsafe { self.slice.get_unchecked(i) })
                .expect("previously added value is missing");
        }
        self.last_start = new_start;
        self.last_end = new_end;
        if self.last_end == 0 {
            return None;
        }
        let cur = unsafe { self.slice.get_unchecked(self.last_end - 1) };
        self.policy.rank(&self.ost, cur)
    }
}

pub type RankWindowAvg<'a, T> = RankWindow<'a, T, f64, RankPolicyAverage>;
pub type RankWindowMin<'a, T> = RankWindow<'a, T, IdxSize, RankPolicyMin>;
pub type RankWindowMax<'a, T> = RankWindow<'a, T, IdxSize, RankPolicyMax>;
pub type RankWindowDense<'a, T> = RankWindow<'a, T, IdxSize, RankPolicyDense>;
pub type RankWindowRandom<'a, T> = RankWindow<'a, T, IdxSize, RankPolicyRandom>;

pub fn rolling_rank<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    params: Option<RollingFnParams>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + num_traits::Num,
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
            values,
            window_size,
            min_periods,
            offset_fn,
            params,
        ),
        RollingRankMethod::Min => rolling_apply_agg_window::<RankWindowMin<T>, _, _, _>(
            values,
            window_size,
            min_periods,
            offset_fn,
            params,
        ),
        RollingRankMethod::Max => rolling_apply_agg_window::<RankWindowMax<T>, _, _, _>(
            values,
            window_size,
            min_periods,
            offset_fn,
            params,
        ),
        RollingRankMethod::Dense => rolling_apply_agg_window::<RankWindowDense<T>, _, _, _>(
            values,
            window_size,
            min_periods,
            offset_fn,
            params,
        ),
        RollingRankMethod::Random => rolling_apply_agg_window::<RankWindowRandom<T>, _, _, _>(
            values,
            window_size,
            min_periods,
            offset_fn,
            params,
        ),
    }
}
