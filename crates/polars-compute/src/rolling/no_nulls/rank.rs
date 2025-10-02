use polars_utils::order_statistic_tree::OrderStatisticTree;
use polars_utils::total_ord::TotalOrd;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::super::rank::*;
use super::*;

#[derive(Debug)]
pub struct RankWindowInner<'a, T>
where
    T: NativeType + TotalOrd,
{
    last_start: usize,
    last_end: usize,
    slice: &'a [T],
    ost: OrderStatisticTree<&'a T>,
}

impl<'a, T> RankWindowInner<'a, T>
where
    T: NativeType,
{
    fn new(
        slice: &'a [T],
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        window_size: Option<usize>,
    ) -> Self {
        let RollingFnParams::Rank { descending, .. } = params.unwrap() else {
            unreachable!("expected RollingFnParams::Rank");
        };
        let cmp = match descending {
            true => |a: &&T, b: &&T| T::tot_cmp(*b, *a),
            false => |a: &&T, b: &&T| T::tot_cmp(*a, *b),
        };
        let ost = OrderStatisticTree::with_capacity(window_size.unwrap(), cmp);
        let mut slf = Self {
            last_start: 0,
            last_end: 0,
            slice,
            ost,
        };
        unsafe {
            RankWindowInner::update_ost(&mut slf, start, end);
        }
        slf
    }

    unsafe fn update_ost(&mut self, new_start: usize, new_end: usize) {
        debug_assert!(self.ost.len() == self.last_end - self.last_start);
        debug_assert!(self.last_start <= self.last_end);
        debug_assert!(self.last_end <= self.slice.len());
        debug_assert!(new_start <= new_end);
        debug_assert!(new_end <= self.slice.len());
        debug_assert!(self.last_start <= new_start);
        debug_assert!(self.last_end <= new_end);

        for i in self.last_start..new_start {
            let v = unsafe { self.slice.get_unchecked(i) };
            self.ost
                .remove(&v)
                .expect("previously added value is missing");
        }
        for i in self.last_end..new_end {
            let v = unsafe { self.slice.get_unchecked(i) };
            self.ost.insert(v);
        }
        self.last_start = new_start;
        self.last_end = new_end;
    }
}

#[derive(Debug)]
struct RankWindowAvg<'a, T>
where
    T: NativeType + TotalOrd,
{
    rw: RankWindowInner<'a, T>,
}

impl<'a, T> RollingAggWindowNoNulls<'a, T, f64> for RankWindowAvg<'a, T>
where
    T: Debug + NativeType + TotalOrd,
{
    fn new(
        slice: &'a [T],
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        window_size: Option<usize>,
    ) -> Self {
        Self {
            rw: RankWindowInner::new(slice, start, end, params, window_size),
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<f64> {
        unsafe {
            self.rw.update_ost(start, end);
        }
        let cur = unsafe { self.rw.slice.get_unchecked(self.rw.last_end - 1) };
        let rank_lo = self.rw.ost.rank_lower(&cur).unwrap() as f64;
        let rank_hi = self.rw.ost.rank_upper(&cur).unwrap() as f64;
        Some((rank_lo + rank_hi) / 2.0)
    }
}

struct RankWindowMinMaxDense<'a, T>
where
    T: NativeType + TotalOrd,
{
    rw: RankWindowInner<'a, T>,
    method: RollingRankMethod,
}

impl<'a, T> RollingAggWindowNoNulls<'a, T, u64> for RankWindowMinMaxDense<'a, T>
where
    T: Debug + NativeType + TotalOrd,
{
    fn new(
        slice: &'a [T],
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        window_size: Option<usize>,
    ) -> Self {
        let Some(RollingFnParams::Rank { method, .. }) = params else {
            unreachable!("expected RollingFnParams::Rank");
        };
        Self {
            rw: RankWindowInner::new(slice, start, end, params, window_size),
            method,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<u64> {
        unsafe {
            self.rw.update_ost(start, end);
        }
        let cur = unsafe { self.rw.slice.get_unchecked(self.rw.last_end - 1) };
        let rank = match self.method {
            RollingRankMethod::Min => self.rw.ost.rank_lower(&cur),
            RollingRankMethod::Max => self.rw.ost.rank_upper(&cur),
            RollingRankMethod::Dense => self.rw.ost.rank_unique(&cur),
            rank => unreachable!("expected Min/Max/Dense rank method, got {rank:?}"),
        };
        Some(rank.unwrap() as u64)
    }
}

#[derive(Debug)]
struct RankWindowRandom<'a, T>
where
    T: NativeType + TotalOrd,
{
    rw: RankWindowInner<'a, T>,
    rng: SmallRng,
}

impl<'a, T> RollingAggWindowNoNulls<'a, T, u64> for RankWindowRandom<'a, T>
where
    T: Debug + NativeType + TotalOrd,
{
    fn new(
        slice: &'a [T],
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        window_size: Option<usize>,
    ) -> Self {
        let RollingFnParams::Rank { seed, .. } = params.unwrap() else {
            unreachable!("expected RollingFnParams::Rank");
        };
        let rng = if let Some(seed) = seed {
            SmallRng::seed_from_u64(seed)
        } else {
            SmallRng::from_os_rng()
        };
        Self {
            rw: RankWindowInner::new(slice, start, end, params, window_size),
            rng,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<u64> {
        unsafe {
            self.rw.update_ost(start, end);
        }
        let cur = unsafe { self.rw.slice.get_unchecked(self.rw.last_end - 1) };
        let rank_lo = self.rw.ost.rank_lower(&cur).unwrap();
        let rank_hi = self.rw.ost.rank_upper(&cur).unwrap();
        Some(self.rng.random_range(rank_lo..rank_hi) as u64)
    }
}

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
        RollingRankMethod::Min | RollingRankMethod::Max | RollingRankMethod::Dense => {
            rolling_apply_agg_window::<RankWindowMinMaxDense<T>, _, _, _>(
                values,
                window_size,
                min_periods,
                offset_fn,
                params,
            )
        },
        RollingRankMethod::Random => rolling_apply_agg_window::<RankWindowRandom<T>, _, _, _>(
            values,
            window_size,
            min_periods,
            offset_fn,
            params,
        ),
    }
}
