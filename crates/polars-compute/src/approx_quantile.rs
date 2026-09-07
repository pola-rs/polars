use std::ops::RangeInclusive;
use std::{fmt, mem};

use either::Either;
pub use kll::KLLSketch;
use polars_utils::total_ord::TotalOrd;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
pub use req::{DoubleReqSketch, ReqSketch};

/// Compute these quantile values using KLL, if the method is Auto.
const KLL_RANGE: RangeInclusive<f64> = 0.05..=0.95;

/// The probability that a query exceeds `error`. KLL calls this `δ`.
///
/// Taken from the 3-sigma rule.
const FAILURE_PROBABILITY: f64 = 1.0 - 0.9973;

/// Looseness of the formal KLL error bound (estimated by measuring).
const KLL_BOUND_LOOSENESS: f64 = 4.6;
/// Looseness of the formal REQ error bound (estimated by measuring).
const REQ_BOUND_LOOSENESS: f64 = 23.0;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ApproxQuantileMethod {
    Auto,
    KLL,
    ReqSketch { hra: bool },
    DoubleReqSketch,
}

pub fn empirical_error_to_formal(empirical_error: f64, method: &ApproxQuantileMethod) -> f64 {
    // The maths get weird at errors close to 1 (especially for REQ), so cap it below that.
    const MAX_ERROR: f64 = 0.90;
    match method {
        ApproxQuantileMethod::Auto => panic!("method not resolved"),
        ApproxQuantileMethod::KLL => f64::min(MAX_ERROR, KLL_BOUND_LOOSENESS * empirical_error),
        ApproxQuantileMethod::ReqSketch { .. } | ApproxQuantileMethod::DoubleReqSketch => {
            f64::min(MAX_ERROR, REQ_BOUND_LOOSENESS * empirical_error)
        },
    }
}

impl ApproxQuantileMethod {
    /// Replace `Auto` by a concrete method.
    pub fn resolve(&self, quantiles: Option<&[f64]>) -> Self {
        use ApproxQuantileMethod as M;
        let M::Auto = self else {
            return self.clone();
        };
        let Some(quantiles) = quantiles else {
            return M::DoubleReqSketch;
        };
        let lo = quantiles.iter().any(|q| *q < *KLL_RANGE.start());
        let hi = quantiles.iter().any(|q| *q > *KLL_RANGE.end());
        match (lo, hi) {
            (false, false) => M::KLL,
            (true, false) => M::ReqSketch { hra: false },
            (false, true) => M::ReqSketch { hra: true },
            (true, true) => M::DoubleReqSketch,
        }
    }
}

#[derive(Debug, Clone)]
struct FinalizedState<T: fmt::Debug + Clone + TotalOrd> {
    /// All retained items, sorted.
    items: Box<[T]>,
    /// Inclusive cumulative weight, i.e. `cum_weight[i]` is the 1-based rank of
    /// `items[i]`. `None` when every item has weight 1.
    cum_weight: Option<Box<[usize]>>,
}

impl<T: fmt::Debug + Clone + TotalOrd> Default for FinalizedState<T> {
    fn default() -> Self {
        Self {
            items: Box::new([]),
            cum_weight: None,
        }
    }
}

impl<T: fmt::Debug + Clone + TotalOrd> FinalizedState<T> {
    fn new(items: Box<[T]>, cum_weight: Option<Box<[usize]>>) -> Self {
        Self { items, cum_weight }
    }

    fn num_items(&self) -> usize {
        match &self.cum_weight {
            Some(cum_weight) => cum_weight.last().map(|x| *x).unwrap_or(0),
            None => self.items.len(),
        }
    }

    /// Merge `other` into `self`.
    ///
    /// This function retains every item of the sketch. No compaction is done
    /// at this point.
    fn merge(&mut self, other: Self) {
        if other.items.is_empty() {
            return;
        }
        if self.items.is_empty() {
            *self = other;
            return;
        }

        let len = self.items.len() + other.items.len();

        if self.cum_weight.is_none() && other.cum_weight.is_none() {
            let items1 = mem::take(&mut self.items).into_vec();
            let items2 = other.items.into_vec();
            let mut items = Vec::with_capacity(len);
            let (i1, i2) = (items1.into_iter(), items2.into_iter());
            merge_sorted(&mut items, i1, i2, TotalOrd::tot_cmp);
            self.items = items.into_boxed_slice();
            return;
        }

        let mut items = Vec::with_capacity(len);
        let mut cum_weight = Vec::with_capacity(len);
        let mut total_weight = 0;
        let mut iter1 = mem::take(self).into_weighted().peekable();
        let mut iter2 = other.into_weighted().peekable();
        while iter1.peek().is_some() || iter2.peek().is_some() {
            let take1 = match (iter1.peek(), iter2.peek()) {
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (Some((x1, _)), Some((x2, _))) => TotalOrd::tot_cmp(x1, x2).is_le(),
                (None, None) => unreachable!(),
            };
            let (item, weight) = match take1 {
                true => iter1.next().unwrap(),
                false => iter2.next().unwrap(),
            };
            total_weight += weight;
            items.push(item);
            cum_weight.push(total_weight);
        }

        debug_assert_eq!(items.len(), len);

        self.items = items.into_boxed_slice();
        self.cum_weight = Some(cum_weight.into_boxed_slice());
    }

    /// Yield every retained item with its own weight.
    fn into_weighted(self) -> impl ExactSizeIterator<Item = (T, usize)> {
        let weights = match self.cum_weight {
            Some(cum_weight) => {
                let mut prev = 0;
                Either::Left(cum_weight.into_vec().into_iter().map(move |cum| {
                    let weight = cum - prev;
                    prev = cum;
                    weight
                }))
            },
            None => Either::Right(std::iter::repeat_n(1, self.items.len())),
        };
        Iterator::zip(self.items.into_vec().into_iter(), weights)
    }

    fn estimate_quantile(&self, quantile: f64) -> Option<&T> {
        assert!(
            (0.0..=1.0).contains(&quantile),
            "quantile should be between 0.0 and 1.0"
        );
        if self.items.is_empty() {
            return None;
        }
        let estimated_rank =
            (quantile * self.num_items().saturating_sub(1) as f64).round() as usize + 1;
        let idx = estimate_quantile_index(self.cum_weight.as_ref(), estimated_rank);
        Some(&self.items[idx])
    }
}

#[inline(never)]
fn estimate_quantile_index(cum_weight: Option<&Box<[usize]>>, estimated_rank: usize) -> usize {
    match cum_weight {
        Some(cum_weight) => cum_weight.partition_point(|w| *w < estimated_rank),
        None => estimated_rank - 1,
    }
}

#[inline(never)]
fn invalid_state() -> ! {
    panic!("invalid state")
}

pub mod kll {
    use super::*;

    /// `CAPACITY_DECAY` specifies how much smaller compactor h+1 is wrt to h.
    /// KLL calls this `c`.
    const CAPACITY_DECAY: f64 = 2.0 / 3.0;

    const MIN_COMPACTOR_SIZE: usize = 2;

    /// Smallest `k` guaranteeing rank error <= `error * n` w.p. >= 1 - `delta` for a
    /// *single* query value, with `delta` = `FAILURE_PROBABILITY`.
    ///
    /// Randomized compaction makes the rank error a zero-mean sum of ±2^h steps, one
    /// per compaction at level `h`. Bounding the compactions per level and summing
    /// the variance over `h < H` gives `std <= (n/k) sqrt(1/(2c-1) + 2/3)`, for
    /// `c > 1/2`. Each step is bounded and mean zero given the levels below it, so
    /// Azuma-Hoeffding turns that into a sub-Gaussian tail with the same proxy,
    /// giving `k = z sqrt(1/(2c-1) + 2/3) / error` for `z = sqrt(2 ln(2/delta))`.
    ///
    /// The bound is computed for the worst case where compactions happen eagerly.
    /// Therefore, the bound is somewhat loose with respect to the implementation.
    fn compute_k(error: f64) -> usize {
        assert!(error > 0.0 && error < 1.0, "invalid error: {error}");

        let z = f64::sqrt(2.0 * f64::ln(2.0 / FAILURE_PROBABILITY)); // sub-Gaussian tail factor for prob. 1 - delta
        let spread = f64::sqrt(1.0 / (2.0 * CAPACITY_DECAY - 1.0) + 2.0 / 3.0); // std bound in units of n/k
        f64::max(MIN_COMPACTOR_SIZE as f64, f64::ceil(z * spread / error)) as usize
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct Level {
        offset: usize,
        size: usize,
        /// Set while the next compaction must take the opposite `coin`.
        mid_pair: bool,
        /// Parity promoted by the previous compaction, see `compact_level`.
        coin: bool,
    }

    #[derive(Debug)]
    struct IngestingState<T: fmt::Debug + Clone + TotalOrd> {
        /// Contents of the compactors. The offsets of the compactors are stored
        /// in the levels vector. The top-level compactor is stored at the start
        /// of this Vec, and the bottom-most compactor is stored at the end of this
        /// Vec.
        ///
        /// This algorithm uses the convention that the top-level compactor has
        /// *level* h-1.  The bottom-level compactor has *level* h,
        /// and height *0*. So the order of `levels` is *reversed* wrt `items`.
        items: Vec<T>,
        levels: Vec<Level>,
        k: usize,
        /// Total number of items that were consumed by this sketch.
        consumed_items: usize,
        /// Maximum number of items before we compact.
        compactor_capacity: usize,
        rng: SmallRng,
        scratch: Vec<T>,
    }

    impl<T: fmt::Debug + Clone + TotalOrd> Clone for IngestingState<T> {
        fn clone(&self) -> Self {
            IngestingState {
                items: self.items.clone(),
                levels: self.levels.clone(),
                k: self.k,
                consumed_items: self.consumed_items,
                compactor_capacity: self.compactor_capacity,
                rng: rand::make_rng(),
                scratch: Vec::new(),
            }
        }
    }

    #[derive(Debug, Clone)]
    enum State<T: fmt::Debug + Clone + TotalOrd> {
        Ingesting(IngestingState<T>),
        Finalized(FinalizedState<T>),
    }

    #[derive(Debug, Clone)]
    #[repr(transparent)]
    pub struct KLLSketch<T: fmt::Debug + Clone + TotalOrd>(State<T>);

    impl<T: fmt::Debug + Clone + TotalOrd> KLLSketch<T> {
        pub fn new(error: f64) -> Self {
            let k = compute_k(error);
            let state = IngestingState {
                items: Vec::new(),
                levels: vec![Level::default()],
                k,
                consumed_items: 0,
                compactor_capacity: k,
                rng: SmallRng::from_rng(&mut rand::rng()),
                scratch: Vec::default(),
            };
            KLLSketch(State::Ingesting(state))
        }

        #[inline]
        pub fn update(&mut self, item: &T) {
            let State::Ingesting(state) = &mut self.0 else {
                invalid_state()
            };
            state.update(item);
        }

        /// Merge the finalized `other` into `self`.
        pub fn merge(&mut self, other: Self) {
            let State::Finalized(other) = other.0 else {
                invalid_state()
            };
            let State::Finalized(state) = &mut self.0 else {
                invalid_state()
            };
            state.merge(other);
        }

        pub fn finalize(&mut self) {
            let placeholder = State::Finalized(FinalizedState::default());
            let state = mem::replace(&mut self.0, placeholder);
            let State::Ingesting(state) = state else {
                invalid_state()
            };
            self.0 = State::Finalized(state.finalize());
        }

        pub fn estimate_quantile(&self, quantile: f64) -> Option<&T> {
            let State::Finalized(state) = &self.0 else {
                invalid_state()
            };
            state.estimate_quantile(quantile)
        }
    }

    impl<T: fmt::Debug + Clone + TotalOrd> IngestingState<T> {
        #[inline]
        pub fn update(&mut self, item: &T) {
            // Fast compare
            if self.items.len() >= self.compactor_capacity {
                self.compact(true);
            }
            self.items.push(item.clone());
            self.levels[0].size += 1;
            self.consumed_items += 1;
        }

        /// Compact all of the compactors from base to top.
        ///
        /// If break_early is true, then the sweeping stops once a compaction has
        /// taken place.
        fn compact(&mut self, break_early: bool) {
            for level in 0..self.levels.len() {
                if self.levels[level].size
                    >= compactor_threshold(self.k, self.levels.len() - 1 - level)
                {
                    if level == self.levels.len() - 1 {
                        self.add_new_compactor();
                    }
                    let old_size = self.items.len();
                    self.compact_level(level);
                    debug_assert!(self.items.len() < old_size);
                    if break_early {
                        break;
                    };
                }
            }
        }

        fn add_new_compactor(&mut self) {
            self.levels.push(Level::default());
            self.compactor_capacity = (0..self.levels.len())
                .map(|level| compactor_threshold(self.k, self.levels.len() - 1 - level))
                .sum();
        }

        fn compact_level(&mut self, level: usize) {
            let mut compact_level = self.levels[level];
            let rand: u8 = self.rng.random();
            let coin1 = rand & 0x1 != 0;

            // Only draw a fresh promotion parity every other compaction, and take
            // the opposite one in between. See DOI 10.3390/s22249612, Sec 3.2.
            compact_level.coin = match compact_level.mid_pair {
                true => !compact_level.coin,
                false => rand & 0x2 != 0,
            };
            compact_level.mid_pair = !compact_level.mid_pair;
            let coin2 = compact_level.coin;

            let mut next_level = self.levels[level + 1];
            let mut compact_start = compact_level.offset;
            let mut compact_end = compact_start + compact_level.size;
            let old_compact_end = compact_end;
            let next_start = next_level.offset;
            let next_end = next_start + next_level.size;
            self.scratch.clear();
            let buf = &mut self.scratch;

            // If there is an odd number of items in this compactor, stash the "straggler" to add it back later
            let mut straggler = None;
            if compact_level.size % 2 != 0 {
                if coin1 {
                    straggler = Some(self.items[compact_start].clone());
                    compact_start += 1;
                } else {
                    straggler = Some(self.items[old_compact_end - 1].clone());
                    compact_end -= 1;
                }
            }

            // The base compactor is not sorted yet
            if level == 0 {
                self.items[compact_start..compact_end].sort_unstable_by(TotalOrd::tot_cmp);
            }

            let next_level_items = self.items[next_start..next_end].iter().cloned();
            let mut compacted_items = self.items[compact_start..compact_end].iter().cloned();

            // Throw away half of the values during the compaction
            if coin2 {
                compacted_items.next();
            }
            let compacted_items = compacted_items.step_by(2);

            // Merge the items into the next compactor
            merge_sorted(buf, next_level_items, compacted_items, TotalOrd::tot_cmp);
            self.items[next_start..next_start + buf.len()].clone_from_slice(&buf);
            next_level.size = buf.len();

            // Add back the straggler
            compact_level.offset = next_level.offset + next_level.size;
            if let Some(item) = straggler {
                self.items[compact_level.offset] = item;
                compact_level.size = 1;
            } else {
                compact_level.size = 0;
            }
            let new_compact_end = compact_level.offset + compact_level.size;

            // Shift all of the compactors below the current one
            let shift = old_compact_end - new_compact_end;
            self.items.drain(new_compact_end..old_compact_end);
            for level_below_compact in self.levels[..level].iter_mut() {
                level_below_compact.offset -= shift;
            }
            self.levels[level] = compact_level;
            self.levels[level + 1] = next_level;

            // Double-check that all the offsets are correct
            let mut offset = 0;
            for level in self.levels.iter().rev() {
                debug_assert_eq!(level.offset, offset);
                offset += level.size;
            }
            debug_assert_eq!(offset, self.items.len());
        }

        fn finalize(self) -> FinalizedState<T> {
            let IngestingState {
                mut items,
                levels,
                consumed_items,
                mut scratch,
                ..
            } = self;

            // Base level is not yet sorted
            let base = levels[0];
            items[base.offset..base.offset + base.size].sort_unstable_by(TotalOrd::tot_cmp);

            // With a single compactor every item has weight 1.
            if levels.len() == 1 {
                return FinalizedState::new(items.into_boxed_slice(), None);
            }

            // Merge all sorted levels
            let level_items: Vec<&[T]> = levels
                .iter()
                .map(|level| &items[level.offset..level.offset + level.size])
                .collect();
            let cum_weights = finalize_merge_levels(&level_items, &mut scratch);

            debug_assert_eq!(scratch.len(), items.len());
            debug_assert_eq!(cum_weights.last().unwrap_or(&0), &consumed_items);

            FinalizedState::new(
                scratch.into_boxed_slice(),
                Some(cum_weights.into_boxed_slice()),
            )
        }
    }

    /// Capacity of the compactor `depth` levels below the top: `ceil(k (2/3)^depth)`
    /// rounded up to an even number, and never below `MIN_COMPACTOR_SIZE`.
    fn compactor_threshold(k: usize, depth: usize) -> usize {
        // Table of 2^63 * (2/3)^i
        const TABLE_SIZE: usize = 64;
        const MUL: [u64; TABLE_SIZE] = {
            let mut result = [0u64; TABLE_SIZE];
            let mut numerator: u128 = 1;
            let mut denominator: u128 = 1;
            let mut i = 0;
            while i < TABLE_SIZE {
                let mut c = 1u128 << 63;
                c *= numerator;
                c /= denominator;
                result[i] = c as u64;
                numerator *= 2;
                denominator *= 3;
                i += 1;
            }
            result
        };
        // Compute ceil(k * 2^i / 3^i) as (k * MUL[i] + (2^63 - 1)) >> 63.
        let nominal_size = (((k as u128) * (MUL[depth] as u128) + (1u128 << 63) - 1) >> 63) as u64;
        debug_assert_eq!(
            nominal_size,
            ((k as u128) * 2u128.pow(depth as u32)).div_ceil(3u128.pow(depth as u32)) as u64
        );
        // Round up to an even number; see the doc comment.
        let nominal_size = usize::try_from(nominal_size).expect("overflow");
        usize::max(nominal_size.next_multiple_of(2), MIN_COMPACTOR_SIZE)
    }
}

pub mod req {
    use super::*;

    /// Stream length to parameterise a fresh sketch for.
    fn initial_n(error: f64) -> usize {
        let k = |n: usize| compute_k(error, FAILURE_PROBABILITY, n);
        let b = |n| compute_b(k(n), n);

        // Choose initial guess of n such that `error * n > 1`: at `error * n ==
        // 1` the `log2` in `compute_k` is zero and `k` overflows.
        let mut n = (f64::ceil(2.0 / error) as usize).next_power_of_two();
        // Ensure that:
        //   1. There are strictly more items than `k`, because at `k == n` no item
        //      could get promoted at all during compaction, and `ReqSketch::new`
        //      relies on `n > k`.
        //   2. The number of consumable items in the sketch is greater than the
        //      capacity of a compactor. Otherwise, we would not even fill up
        //      that first compactor.
        let n_is_ok = |n| n > k(n) && n > b(n);
        while !n_is_ok(n) {
            n = n.checked_mul(2).expect("no sketch size fits this error");
        }
        // TODO: [amber] Consider an additional factor of 8 or smth.
        n
    }

    fn compute_k(error: f64, failure_prob: f64, n: usize) -> usize {
        assert!(error > 0.0 && error < 1.0, "invalid error: {error}");
        assert!(
            failure_prob > 0.0 && failure_prob <= 0.5,
            "invalid failure probability: {failure_prob}"
        );

        // Eq. 6
        let k = 2 * f64::ceil(
            (4.0 / error) * f64::sqrt((-f64::ln(failure_prob)) / f64::log2(error * n as f64)),
        ) as usize;
        assert!(k >= 2);
        k
    }

    fn compute_b(k: usize, n: usize) -> usize {
        // Sec 2.1: k is an *even* integer parameter.
        assert!(k > 0 && k % 2 == 0, "k must be a positive even integer");
        2 * k * (usize::div_ceil(n, k) * 2 - 1).ilog2() as usize
    }

    /// Put the items a compaction takes at the front of the
    /// compactor, next to the level they are promoted into.
    #[inline(always)]
    fn cmp_desc<const HRA: bool, T: TotalOrd>(a: &T, b: &T) -> std::cmp::Ordering {
        match HRA {
            false => TotalOrd::tot_cmp(b, a),
            true => TotalOrd::tot_cmp(a, b),
        }
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct Level {
        offset: usize,
        size: usize,
        compaction_schedule: u64,
        /// Parity promoted by the previous compaction, see `compact_level_once`.
        coin: bool,
    }

    #[derive(Debug)]
    struct IngestingState<T: fmt::Debug + Clone + TotalOrd> {
        /// Contents of the relative compactors. The offsets of the compactors
        /// are stored in the levels vector. The top-level compactor is stored at
        /// the start of this Vec, and the bottom-most compactor is stored at the
        /// end of this Vec.
        ///
        /// This algorithm uses the convention that the top-level compactor has
        /// *level* h-1.  The bottom-level compactor has *level* h,
        /// and height *0*. So the order of `levels` is *reversed* wrt `items`.
        items: Vec<T>,
        /// Scratch Vec to reduce an allocation during merging.
        scratch: Vec<T>,
        levels: Vec<Level>,
        /// Bit that specifies if this sketch is high-rank-accurate or low-rank-accurate.
        is_hra: bool,
        /// Upper bound on the number of items this sketch is parameterised
        /// for. Squared on every growth.
        n: usize,
        /// The allowed error as a fraction of `n`.
        error: f64,
        /// k parameter of the paper: the size of a compactor section. Impacts
        /// how many items are protected during a compaction. Shrinks over time,
        /// see `ensure_enough_sections`.
        k: usize,
        consumed_items: usize,
        rng: SmallRng,
    }

    impl<T: fmt::Debug + Clone + TotalOrd> Clone for IngestingState<T> {
        fn clone(&self) -> Self {
            IngestingState {
                items: self.items.clone(),
                scratch: Vec::new(),
                levels: self.levels.clone(),
                is_hra: self.is_hra,
                n: self.n,
                error: self.error,
                k: self.k,
                consumed_items: self.consumed_items,
                rng: rand::make_rng(),
            }
        }
    }

    #[derive(Debug, Clone)]
    enum State<T: fmt::Debug + Clone + TotalOrd> {
        Ingesting(IngestingState<T>),
        Finalized(FinalizedState<T>),
    }

    #[derive(Debug, Clone)]
    pub struct ReqSketch<T: fmt::Debug + Clone + TotalOrd> {
        state: State<T>,
        is_hra: bool,
    }

    impl<T: fmt::Debug + Clone + TotalOrd> ReqSketch<T> {
        pub fn new(error: f64, hra: bool) -> Self {
            let n = initial_n(error);
            let k = compute_k(error, FAILURE_PROBABILITY, n);
            assert!(n > k, "n must be greater than k");
            let state = IngestingState {
                items: Vec::new(),
                scratch: Vec::new(),
                levels: vec![Level::default()],
                is_hra: hra,
                n,
                error,
                k,
                consumed_items: 0,
                rng: rand::make_rng(),
            };
            ReqSketch {
                state: State::Ingesting(state),
                is_hra: hra,
            }
        }

        #[inline]
        pub fn num_items(&self) -> usize {
            match &self.state {
                State::Ingesting(state) => state.consumed_items,
                State::Finalized(state) => state.num_items(),
            }
        }

        #[inline]
        pub fn update(&mut self, item: &T) {
            let State::Ingesting(state) = &mut self.state else {
                invalid_state()
            };
            state.update(item);
        }

        /// Merge the finalized `other` into `self`.
        pub fn merge(&mut self, other: Self) {
            assert_eq!(self.is_hra, other.is_hra);
            let State::Finalized(other) = other.state else {
                invalid_state()
            };
            let State::Finalized(state) = &mut self.state else {
                invalid_state()
            };
            state.merge(other);
        }

        #[inline]
        pub fn finalize(&mut self) {
            let placeholder = State::Finalized(FinalizedState::default());
            let state = mem::replace(&mut self.state, placeholder);
            let State::Ingesting(state) = state else {
                invalid_state()
            };
            self.state = State::Finalized(state.finalize());
        }

        #[inline]
        pub fn estimate_quantile(&self, quantile: f64) -> Option<&T> {
            let State::Finalized(state) = &self.state else {
                invalid_state()
            };
            state.estimate_quantile(quantile)
        }
    }

    /// A pair of [`ReqSketch`]es that is relative-error accurate over the
    /// *whole* rank range.
    ///
    /// Costs 2x the size and speed of a single [`ReqSketch`]
    #[derive(Debug, Clone)]
    pub struct DoubleReqSketch<T: fmt::Debug + Clone + TotalOrd> {
        lra: ReqSketch<T>,
        hra: ReqSketch<T>,
    }

    impl<T: fmt::Debug + Clone + TotalOrd> DoubleReqSketch<T> {
        pub fn new(error: f64) -> Self {
            DoubleReqSketch {
                lra: ReqSketch::new(error, false),
                hra: ReqSketch::new(error, true),
            }
        }

        #[inline]
        pub fn update(&mut self, item: &T) {
            self.lra.update(item);
            self.hra.update(item);
        }

        pub fn merge(&mut self, other: Self) {
            self.lra.merge(other.lra);
            self.hra.merge(other.hra);
        }

        pub fn finalize(&mut self) {
            self.lra.finalize();
            self.hra.finalize();
        }

        pub fn num_items(&self) -> usize {
            debug_assert_eq!(self.lra.num_items(), self.hra.num_items());
            self.lra.num_items()
        }

        pub fn estimate_quantile(&self, quantile: f64) -> Option<&T> {
            match quantile <= 0.5 {
                true => self.lra.estimate_quantile(quantile),
                false => self.hra.estimate_quantile(quantile),
            }
        }
    }

    impl<T: fmt::Debug + Clone + TotalOrd> IngestingState<T> {
        #[inline]
        pub fn update(&mut self, item: &T) {
            self.compact_if_needed(0);
            self.items.push(item.clone());
            self.levels[0].size += 1;
            self.consumed_items += 1;
        }

        /// Grow the compactors once a compaction schedule runs out of sections.
        fn close_out_if_needed(&mut self, level: usize) {
            let num_sections = self.num_sections();
            if num_sections >= 64 {
                // We assume that the compaction schedule will never overflow over 64 bits.
                return;
            }

            let schedule = self.levels[level].compaction_schedule;
            let sections_needed = u64::BITS - schedule.leading_zeros();
            if sections_needed < num_sections as u32 {
                return;
            }

            // The paper squares here, but growing is quite cheap in practice,
            // so we just amortize by doubling.
            while sections_needed >= self.num_sections() as u32 {
                self.n = self.n.checked_mul(2).expect("overflow");
                self.k = compute_k(self.error, FAILURE_PROBABILITY, self.n);
            }
        }

        /// `B` of the paper: the capacity of every relative compactor.
        fn compactor_capacity(&self) -> usize {
            compute_b(self.k, self.n)
        }

        /// The largest number of sections a single compaction may cover.
        fn num_sections(&self) -> usize {
            self.compactor_capacity() / (2 * self.k)
        }

        fn is_compactor_full(&self, level: usize) -> bool {
            self.levels[level].size >= self.compactor_capacity()
        }

        /// Compact all of the compactors from base to top.
        fn compact_if_needed(&mut self, level: usize) {
            if self.is_compactor_full(level) {
                let old_size = self.levels[level].size;
                self.compact_level_once(level);
                debug_assert!(self.levels[level].size < old_size);
            }
            debug_assert!(!self.is_compactor_full(level))
        }

        fn add_new_compactor(&mut self) {
            self.levels.push(Level::default());
        }

        fn compact_level_once(&mut self, level: usize) {
            if level == self.levels.len() - 1 {
                self.add_new_compactor();
            }
            debug_assert!(
                self.levels[level].size >= self.compactor_capacity(),
                "compactor is not full"
            );
            debug_assert_eq!(
                self.levels[level + 1].offset + self.levels[level + 1].size,
                self.levels[level].offset
            );

            let compactor_start = self.levels[level].offset;
            let compactor_size = self.levels[level].size;
            let compactor_end = compactor_start + compactor_size;
            let compactor = &self.items[compactor_start..compactor_end];

            if level > 0 {
                debug_assert!(
                    compactor.windows(2).all(|w| match self.is_hra {
                        false => cmp_desc::<false, T>(&w[0], &w[1]).is_le(),
                        true => cmp_desc::<true, T>(&w[0], &w[1]).is_le(),
                    }),
                    "compactor is not a single descending run"
                );
            }

            let z_c = self.levels[level].compaction_schedule.trailing_ones();
            let l_c = usize::min(z_c as usize + 1, self.num_sections()) * self.k;
            let promote_count = compactor[self.compactor_capacity() - l_c..].len() & !1;
            debug_assert!(l_c <= self.compactor_capacity() / 2);
            debug_assert!(l_c % 2 == 0);
            debug_assert!(promote_count >= l_c);

            // Only draw a fresh promotion parity every other compaction, and take
            // the opposite one in between. See DOI 10.3390/s22249612, Sec 3.2.
            let coin = match self.levels[level].compaction_schedule % 2 != 0 {
                true => !self.levels[level].coin,
                false => self.rng.random(),
            };
            self.levels[level].coin = coin;

            // Level 0 is not sorted yet.
            let compactor = &mut self.items[compactor_start..compactor_end];
            if level == 0 {
                match self.is_hra {
                    false => Self::partition_compactor::<false>(compactor, promote_count),
                    true => Self::partition_compactor::<true>(compactor, promote_count),
                }
            }

            // Throw away half of the values during the compaction, gathering the
            // survivors at the front of the compacted range.
            for i in 0..promote_count / 2 {
                compactor.swap(i, 2 * i + coin as usize);
            }

            // Drop the non-promoted items from the item pool.
            let gap_start = compactor_start + promote_count / 2;
            let gap_end = compactor_start + promote_count;
            self.items.drain(gap_start..gap_end);

            // Merge the promoted items into the next compactor.
            let next_start = self.levels[level + 1].offset;
            let next_split = self.levels[level + 1].size;
            let next_end = next_split + promote_count / 2;
            if next_split > 0 {
                let next = next_start..next_start + next_end;
                let (left, right) = self.items[next.clone()].split_at(next_split);
                let (left, right) = (left.iter().cloned(), right.iter().cloned());
                self.scratch.clear();
                match self.is_hra {
                    false => merge_sorted(&mut self.scratch, left, right, cmp_desc::<false, T>),
                    true => merge_sorted(&mut self.scratch, left, right, cmp_desc::<true, T>),
                }
                self.items[next].clone_from_slice(&self.scratch);
            }
            self.levels[level + 1].size = next_end;
            self.levels[level].offset += promote_count / 2;
            self.levels[level].size -= promote_count;

            // Update the other compactor offsets after removing the non-promoted items.
            for level_below_compact in self.levels[..level].iter_mut() {
                level_below_compact.offset -= promote_count / 2;
            }

            // Double-check that all the offsets are correct
            let mut offset = 0;
            for level in self.levels.iter().rev() {
                debug_assert_eq!(level.offset, offset);
                offset += level.size;
            }
            debug_assert_eq!(offset, self.items.len());

            self.levels[level].compaction_schedule += 1;
            self.close_out_if_needed(level);
            self.compact_if_needed(level + 1);
        }

        fn partition_compactor<const HRA: bool>(compactor: &mut [T], promote_count: usize) {
            compactor.select_nth_unstable_by(promote_count, cmp_desc::<HRA, T>);
            compactor[..promote_count].sort_unstable_by(cmp_desc::<HRA, T>);
        }

        fn finalize(self) -> FinalizedState<T> {
            let IngestingState {
                mut items,
                levels,
                consumed_items,
                mut scratch,
                ..
            } = self;

            // Compaction only partially orders a compactor, so sort them all.
            for level in levels.iter() {
                items[level.offset..level.offset + level.size].sort_unstable_by(TotalOrd::tot_cmp);
            }

            // With a single compactor every item has weight 1.
            if levels.len() == 1 {
                return FinalizedState::new(items.into_boxed_slice(), None);
            }

            // Merge all sorted levels
            let level_items: Vec<&[T]> = levels
                .iter()
                .map(|level| &items[level.offset..level.offset + level.size])
                .collect();
            let cum_weights = finalize_merge_levels(&level_items, &mut scratch);

            debug_assert_eq!(scratch.len(), items.len());
            debug_assert_eq!(cum_weights.last().unwrap_or(&0), &consumed_items);

            FinalizedState::new(
                scratch.into_boxed_slice(),
                Some(cum_weights.into_boxed_slice()),
            )
        }
    }
}

/// H-way merge-sort of the per-level sorted runs into a single sorted run, and
/// the inclusive cumulative weight of every merged item.
///
/// `levels[i]` holds the items of level `i`, each standing for `2^i` ingested
/// items, so the last cumulative weight is the total number of ingested items.
///
/// The merged items are written into `out`, which is cleared first so that
/// callers can hand over a scratch buffer.
fn finalize_merge_levels<T: fmt::Debug + Clone + TotalOrd>(
    levels: &[&[T]],
    out: &mut Vec<T>,
) -> Vec<usize> {
    let num_items: usize = levels.iter().map(|level| level.len()).sum();
    out.clear();
    out.reserve_exact(num_items);
    let mut cum_weights = Vec::with_capacity(num_items);
    let mut cursors: Vec<usize> = vec![0; levels.len()];

    // Are we done draining this level?
    let is_done = |level: usize, cursors: &[usize]| cursors[level] >= levels[level].len();
    // Get the next value corresponding to level `level`.
    let next_value = |level: usize, cursors: &[usize]| &levels[level][cursors[level]];

    while let Some(level_idx) = (0..levels.len())
        .filter(|i| !is_done(*i, &cursors))
        .min_by(|i1, i2| TotalOrd::tot_cmp(next_value(*i1, &cursors), next_value(*i2, &cursors)))
    {
        let weight = 1usize << level_idx;
        let cum_weight = cum_weights.last().unwrap_or(&0) + weight;
        out.push(next_value(level_idx, &cursors).clone());
        cum_weights.push(cum_weight);
        cursors[level_idx] += 1;
    }

    debug_assert_eq!(out.len(), num_items);
    debug_assert_eq!(cum_weights.len(), num_items);
    cum_weights
}

/// Append the merge of two runs, both sorted by `compare`, to `vec`.
#[inline(never)]
fn merge_sorted<T>(
    vec: &mut Vec<T>,
    iter1: impl ExactSizeIterator<Item = T>,
    iter2: impl ExactSizeIterator<Item = T>,
    mut compare: impl FnMut(&T, &T) -> std::cmp::Ordering,
) {
    vec.reserve(iter1.len() + iter2.len());
    let mut iter1 = iter1.peekable();
    let mut iter2 = iter2.peekable();
    loop {
        match (iter1.peek(), iter2.peek()) {
            (None, None) => return,
            (Some(_), None) => vec.push(iter1.next().unwrap()),
            (None, Some(_)) => vec.push(iter2.next().unwrap()),
            (Some(x1), Some(x2)) => {
                if compare(x1, x2).is_le() {
                    vec.push(iter1.next().unwrap());
                } else {
                    vec.push(iter2.next().unwrap())
                }
            },
        }
    }
}

/// A sketch picked by [`ApproxQuantileMethod`].
#[derive(Debug, Clone)]
pub enum Sketch<T: fmt::Debug + Clone + TotalOrd> {
    Kll(KLLSketch<T>),
    Req(ReqSketch<T>),
    DoubleReq(DoubleReqSketch<T>),
}

impl<T: fmt::Debug + Clone + TotalOrd> Sketch<T> {
    pub fn new(method: &ApproxQuantileMethod, error: f64) -> Self {
        match method {
            ApproxQuantileMethod::Auto => unreachable!(),
            ApproxQuantileMethod::KLL => Sketch::Kll(KLLSketch::new(error)),
            ApproxQuantileMethod::ReqSketch { hra } => Sketch::Req(ReqSketch::new(error, *hra)),
            ApproxQuantileMethod::DoubleReqSketch => Sketch::DoubleReq(DoubleReqSketch::new(error)),
        }
    }

    #[inline]
    pub fn update(&mut self, item: &T) {
        match self {
            Sketch::Kll(s) => s.update(item),
            Sketch::Req(s) => s.update(item),
            Sketch::DoubleReq(s) => s.update(item),
        }
    }

    /// Merge the finalized `other` into `self`.
    pub fn merge(&mut self, other: Self) {
        match (self, other) {
            (Sketch::Kll(a), Sketch::Kll(b)) => a.merge(b),
            (Sketch::Req(a), Sketch::Req(b)) => a.merge(b),
            (Sketch::DoubleReq(a), Sketch::DoubleReq(b)) => a.merge(b),
            _ => panic!("cannot merge sketches of a different method"),
        }
    }

    pub fn finalize(&mut self) {
        match self {
            Sketch::Kll(s) => s.finalize(),
            Sketch::Req(s) => s.finalize(),
            Sketch::DoubleReq(s) => s.finalize(),
        }
    }

    pub fn estimate_quantile(&self, quantile: f64) -> Option<&T> {
        match self {
            Sketch::Kll(s) => s.estimate_quantile(quantile),
            Sketch::Req(s) => s.estimate_quantile(quantile),
            Sketch::DoubleReq(s) => s.estimate_quantile(quantile),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::kll::KLLSketch;
    use super::req::ReqSketch;
    use super::{ApproxQuantileMethod, Sketch};

    #[test]
    fn auto_resolves_by_queried_quantiles() {
        use ApproxQuantileMethod as M;
        let auto = |qs: Option<&[f64]>| M::Auto.resolve(qs);

        assert_eq!(auto(None), M::DoubleReqSketch);
        assert_eq!(auto(Some(&[])), M::KLL);
        assert_eq!(auto(Some(&[0.5])), M::KLL);
        // The bounds themselves are in range.
        assert_eq!(auto(Some(&[0.05, 0.95])), M::KLL);
        assert_eq!(auto(Some(&[0.01])), M::ReqSketch { hra: false });
        assert_eq!(auto(Some(&[0.99])), M::ReqSketch { hra: true });
        assert_eq!(auto(Some(&[0.0, 0.5])), M::ReqSketch { hra: false });
        assert_eq!(auto(Some(&[0.5, 1.0])), M::ReqSketch { hra: true });
        assert_eq!(auto(Some(&[0.01, 0.99])), M::DoubleReqSketch);

        // An explicit method is never overridden.
        for method in [M::KLL, M::ReqSketch { hra: false }, M::DoubleReqSketch] {
            assert_eq!(method.resolve(None), method);
            assert_eq!(method.resolve(Some(&[0.01, 0.99])), method);
        }
    }

    /// Deterministic shuffle of `0..n`, so a partition is not a contiguous range.
    fn shuffled(n: usize) -> Vec<f64> {
        (0..n).map(|i| ((i * 7919) % n) as f64).collect()
    }

    /// Build one sketch per partition and merge them into one.
    fn merged(
        method: &ApproxQuantileMethod,
        error: f64,
        data: &[f64],
        parts: usize,
    ) -> Sketch<f64> {
        let mut sketches: Vec<_> = data
            .chunks(usize::div_ceil(data.len(), parts))
            .map(|chunk| {
                let mut sketch = Sketch::new(method, error);
                for v in chunk {
                    sketch.update(v);
                }
                sketch.finalize();
                sketch
            })
            .collect();
        let mut sketch = sketches.remove(0);
        for other in sketches {
            sketch.merge(other);
        }
        sketch
    }

    /// Ingest `data` into a fresh finalized sketch.
    fn finalized(method: &ApproxQuantileMethod, error: f64, data: &[f64]) -> Sketch<f64> {
        let mut sketch = Sketch::new(method, error);
        for v in data {
            sketch.update(v);
        }
        sketch.finalize();
        sketch
    }

    const METHODS: [ApproxQuantileMethod; 4] = [
        ApproxQuantileMethod::KLL,
        ApproxQuantileMethod::ReqSketch { hra: false },
        ApproxQuantileMethod::ReqSketch { hra: true },
        ApproxQuantileMethod::DoubleReqSketch,
    ];

    /// A merged sketch answers within the same rank error as a single one.
    #[test]
    fn merge_keeps_rank_error() {
        const N: usize = 100_000;
        const ERROR: f64 = 0.01;
        const QUANTILES: [f64; 7] = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];

        let data = shuffled(N);
        for method in &METHODS {
            for parts in [2, 8, 24] {
                let sketch = merged(method, ERROR, &data, parts);
                for q in QUANTILES {
                    // The data is `0..N` shuffled, so a value *is* its own rank.
                    let got = *sketch.estimate_quantile(q).unwrap();
                    let want = q * (N - 1) as f64;
                    assert!(
                        (got - want).abs() <= ERROR * N as f64,
                        "{method:?} with {parts} parts at q={q}: got rank {got}, want {want}",
                    );
                }
            }
        }
    }

    /// Merging in empty sketches must not disturb the estimates, in either
    /// direction.
    #[test]
    fn merge_of_empty_is_neutral() {
        const N: usize = 20_000;
        const ERROR: f64 = 0.01;

        let data = shuffled(N);
        for method in &METHODS {
            let empty = || finalized(method, ERROR, &[]);
            let full = || finalized(method, ERROR, &data);

            let mut into_full = full();
            into_full.merge(empty());
            let mut into_empty = empty();
            into_empty.merge(full());

            for (name, sketch) in [("full+empty", &into_full), ("empty+full", &into_empty)] {
                for q in [0.0, 0.5, 1.0] {
                    let got = *sketch.estimate_quantile(q).unwrap();
                    let want = q * (N - 1) as f64;
                    assert!(
                        (got - want).abs() <= ERROR * N as f64,
                        "{method:?} {name} at q={q}: got rank {got}, want {want}",
                    );
                }
            }
        }
    }

    /// Parts small enough never to compact keep every item at weight 1, so
    /// their merge is exact.
    #[test]
    fn merge_of_unweighted_parts_is_exact() {
        const N: usize = 600;
        const ERROR: f64 = 0.01;

        let data = shuffled(N);
        for method in &METHODS {
            let sketch = merged(method, ERROR, &data, 6);
            for q in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let got = *sketch.estimate_quantile(q).unwrap();
                let want = (q * (N - 1) as f64).round();
                assert_eq!(got, want, "{method:?} at q={q}");
            }
        }
    }

    /// A weighted sketch merged with an unweighted one.
    #[test]
    fn merge_of_mixed_weights() {
        const BIG: usize = 50_000;
        const SMALL: usize = 100;
        const ERROR: f64 = 0.01;

        let data = shuffled(BIG + SMALL);
        for method in &METHODS {
            let mut sketch = finalized(method, ERROR, &data[..BIG]);
            sketch.merge(finalized(method, ERROR, &data[BIG..]));

            for q in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let got = *sketch.estimate_quantile(q).unwrap();
                let want = q * (BIG + SMALL - 1) as f64;
                assert!(
                    (got - want).abs() <= ERROR * (BIG + SMALL) as f64,
                    "{method:?} at q={q}: got rank {got}, want {want}",
                );
            }
        }
    }

    /// Every `error` in the usable range has to produce a working sketch, and a
    /// smaller `error` must never give a worse answer. `initial_n` used to leave
    /// `n == k` for some errors (0.3 among them), which tripped `ReqSketch::new`.
    #[test]
    fn error_sweep_is_sound() {
        const N: usize = 50_000;
        const ERRORS: [f64; 14] = [
            0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01, 0.005,
        ];

        let data = shuffled(N);
        for method in &METHODS {
            for error in ERRORS {
                let mut sketch = Sketch::new(method, error);
                for v in &data {
                    sketch.update(v);
                }
                sketch.finalize();
                for q in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    let got = *sketch.estimate_quantile(q).unwrap();
                    let want = q * (N - 1) as f64;
                    assert!(
                        (got - want).abs() <= error * N as f64,
                        "{method:?} at error={error} q={q}: got rank {got}, want {want}",
                    );
                }
            }
        }
    }

    /// Clones must not make identical random choices.
    #[test]
    fn clones_are_reseeded() {
        const QUANTILES: [f64; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];
        let data: Vec<f64> = (0..20_000).map(|i| ((i * 7919) % 20_000) as f64).collect();

        macro_rules! assert_diverges {
            ($name:literal, $new:expr) => {{
                let agreed = (0..10)
                    .filter(|_| {
                        let mut base = $new;
                        for v in &data[..5_000] {
                            base.update(v);
                        }
                        let (mut a, mut b) = (base.clone(), base.clone());
                        for v in &data[5_000..] {
                            a.update(v);
                            b.update(v);
                        }
                        a.finalize();
                        b.finalize();
                        QUANTILES
                            .iter()
                            .all(|q| a.estimate_quantile(*q) == b.estimate_quantile(*q))
                    })
                    .count();
                assert!(agreed <= 2, "{} clones agreed {agreed}/10 times", $name);
            }};
        }

        assert_diverges!("ReqSketch", ReqSketch::new(0.01, true));
        assert_diverges!("KLLSketch", KLLSketch::new(0.01));
    }
}
