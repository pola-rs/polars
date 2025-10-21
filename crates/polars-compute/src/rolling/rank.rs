use std::fmt::Debug;

use polars_utils::IdxSize;
use polars_utils::order_statistic_tree::OrderStatisticTree;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::*;

pub trait RankPolicy<T, Out>: Debug
where
    T: NativeType,
    Out: NativeType,
{
    fn new(params: &RollingFnParams) -> Self;
    fn rank<'a>(&mut self, ost: &OrderStatisticTree<&'a T>, value: &'a T) -> Option<Out>;
}

#[derive(Debug)]
pub struct RankPolicyAverage;

impl<T: NativeType> RankPolicy<T, f64> for RankPolicyAverage {
    fn new(_params: &RollingFnParams) -> Self {
        Self
    }
    fn rank<'a>(&mut self, ost: &OrderStatisticTree<&'a T>, value: &'a T) -> Option<f64> {
        let rank_range = ost.rank_range(&value).ok()?;
        let rank_lo = (rank_range.start() + 1) as f64;
        let rank_hi = (rank_range.end() + 1) as f64;
        Some((rank_lo + rank_hi) / 2.0)
    }
}

#[derive(Debug)]
pub struct RankPolicyMin;

impl<T: NativeType> RankPolicy<T, IdxSize> for RankPolicyMin {
    fn new(_params: &RollingFnParams) -> Self {
        Self
    }
    fn rank<'a>(&mut self, ost: &OrderStatisticTree<&'a T>, value: &'a T) -> Option<IdxSize> {
        Some(IdxSize::try_from(ost.rank_range(&value).ok()?.start() + 1).unwrap())
    }
}

#[derive(Debug)]
pub struct RankPolicyMax;

impl<T: NativeType> RankPolicy<T, IdxSize> for RankPolicyMax {
    fn new(_params: &RollingFnParams) -> Self {
        Self
    }
    fn rank<'a>(&mut self, ost: &OrderStatisticTree<&'a T>, value: &'a T) -> Option<IdxSize> {
        Some(IdxSize::try_from(*ost.rank_range(&value).ok()?.end() + 1).unwrap())
    }
}

#[derive(Debug)]
pub struct RankPolicyDense;

impl<T: NativeType> RankPolicy<T, IdxSize> for RankPolicyDense {
    fn new(_params: &RollingFnParams) -> Self {
        Self
    }
    fn rank<'a>(&mut self, ost: &OrderStatisticTree<&'a T>, value: &'a T) -> Option<IdxSize> {
        Some(IdxSize::try_from(ost.rank_unique(&value).ok()? + 1).unwrap())
    }
}

#[derive(Debug)]
pub struct RankPolicyRandom {
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
        let rank_range = ost.rank_range(&value).ok()?;
        let rank_lo = rank_range.start() + 1;
        let rank_hi = rank_range.end() + 1;
        Some(IdxSize::try_from(self.rng.random_range(rank_lo..=rank_hi)).unwrap())
    }
}
