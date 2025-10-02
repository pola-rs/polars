use std::fmt::Debug;

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
        let rank_lo = ost.rank_lower(&value).ok()? as f64;
        let rank_hi = ost.rank_upper(&value).ok()? as f64;
        Some((rank_lo + rank_hi) / 2.0)
    }
}

#[derive(Debug)]
pub(super) struct RankPolicyMin;

impl<T: NativeType> RankPolicy<T, u64> for RankPolicyMin {
    fn new(_params: &RollingFnParams) -> Self {
        Self
    }
    fn rank<'a>(&mut self, ost: &OrderStatisticTree<&'a T>, value: &'a T) -> Option<u64> {
        Some(ost.rank_lower(&value).ok()? as u64)
    }
}

#[derive(Debug)]
pub(super) struct RankPolicyMax;

impl<T: NativeType> RankPolicy<T, u64> for RankPolicyMax {
    fn new(_params: &RollingFnParams) -> Self {
        Self
    }
    fn rank<'a>(&mut self, ost: &OrderStatisticTree<&'a T>, value: &'a T) -> Option<u64> {
        Some(ost.rank_upper(&value).ok()? as u64)
    }
}

#[derive(Debug)]
pub(super) struct RankPolicyDense;

impl<T: NativeType> RankPolicy<T, u64> for RankPolicyDense {
    fn new(_params: &RollingFnParams) -> Self {
        Self
    }
    fn rank<'a>(&mut self, ost: &OrderStatisticTree<&'a T>, value: &'a T) -> Option<u64> {
        Some(ost.rank_unique(&value).ok()? as u64)
    }
}

#[derive(Debug)]
pub(super) struct RankPolicyRandom {
    rng: SmallRng,
}

impl<T: NativeType> RankPolicy<T, u64> for RankPolicyRandom {
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
    fn rank<'a>(&mut self, ost: &OrderStatisticTree<&'a T>, value: &'a T) -> Option<u64> {
        let rank_lo = ost.rank_lower(&value).ok()?;
        let rank_hi = ost.rank_upper(&value).ok()?;
        Some(self.rng.random_range(rank_lo..=rank_hi) as u64)
    }
}
