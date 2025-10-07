use std::fmt::Debug;

use polars_utils::IdxSize;
use polars_utils::order_statistic_tree::OrderStatisticTree;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::*;

pub(super) trait RankPolicy<T, Out>: Debug
where
    T: NativeType,
    Out: NativeType,
{
    fn new(params: &RollingFnParams) -> Self;
    fn rank<'a>(&mut self, ost: &OrderStatisticTree<&'a T>, value: &'a T) -> Option<Out>;
}

#[derive(Debug)]
pub(super) struct RankPolicyAverage;

impl<T: NativeType> RankPolicy<T, f64> for RankPolicyAverage {
    fn new(_params: &RollingFnParams) -> Self {
        Self
    }
    fn rank<'a>(&mut self, ost: &OrderStatisticTree<&'a T>, value: &'a T) -> Option<f64> {
        let rank_lo = (ost.rank_lower(&value).ok()? + 1) as f64;
        let rank_hi = (ost.rank_upper(&value).ok()? + 1) as f64;
        Some((rank_lo + rank_hi) / 2.0)
    }
}

#[derive(Debug)]
pub(super) struct RankPolicyMin;

impl<T: NativeType> RankPolicy<T, IdxSize> for RankPolicyMin {
    fn new(_params: &RollingFnParams) -> Self {
        Self
    }
    fn rank<'a>(&mut self, ost: &OrderStatisticTree<&'a T>, value: &'a T) -> Option<IdxSize> {
        Some(IdxSize::try_from(ost.rank_lower(&value).ok()? + 1).unwrap())
    }
}

#[derive(Debug)]
pub(super) struct RankPolicyMax;

impl<T: NativeType> RankPolicy<T, IdxSize> for RankPolicyMax {
    fn new(_params: &RollingFnParams) -> Self {
        Self
    }
    fn rank<'a>(&mut self, ost: &OrderStatisticTree<&'a T>, value: &'a T) -> Option<IdxSize> {
        Some(IdxSize::try_from(ost.rank_upper(&value).ok()? + 1).unwrap())
    }
}

#[derive(Debug)]
pub(super) struct RankPolicyDense;

impl<T: NativeType> RankPolicy<T, IdxSize> for RankPolicyDense {
    fn new(_params: &RollingFnParams) -> Self {
        Self
    }
    fn rank<'a>(&mut self, ost: &OrderStatisticTree<&'a T>, value: &'a T) -> Option<IdxSize> {
        Some(IdxSize::try_from(ost.rank_unique(&value).ok()? + 1).unwrap())
    }
}

#[derive(Debug)]
pub(super) struct RankPolicyRandom {
    rng: SmallRng,
}

impl<T: NativeType> RankPolicy<T, IdxSize> for RankPolicyRandom {
    fn new(params: &RollingFnParams) -> Self {
        let RollingFnParams::Rank { seed, .. } = params else {
            unreachable!("expected RollingFnParams::Rank");
        };
        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(*s),
            None => SmallRng::from_os_rng(),
        };
        Self { rng }
    }
    fn rank<'a>(&mut self, ost: &OrderStatisticTree<&'a T>, value: &'a T) -> Option<IdxSize> {
        let rank_lo = ost.rank_lower(&value).ok()? + 1;
        let rank_hi = ost.rank_upper(&value).ok()? + 1;
        Some(IdxSize::try_from(self.rng.random_range(rank_lo..=rank_hi)).unwrap())
    }
}
