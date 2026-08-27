use std::ops::RangeInclusive;
use std::{fmt, mem};

pub use kll::KLLSketch;
use polars_utils::total_ord::TotalOrd;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
pub use req::{DoubleReqSketch, ReqSketch};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ApproxQuantileMethod {
    Auto,
    KLL,
    ReqSketch { hra: bool },
    DoubleReqSketch,
}

/// Quantiles in this range are served well enough by KLL's uniform error.
const KLL_RANGE: RangeInclusive<f64> = 0.05..=0.95;

impl ApproxQuantileMethod {
    /// Replace `Auto` by a concrete method. Set `quantiles` to `None` if the
    /// quantiles are not known at plan time.
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

    fn estimate_rank(&self, value: &T) -> usize {
        todo!()
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

    // [amber]
    // * In experiments I benchmarked that a Vec<Vec<T>> approach is slower than
    //   having a single `items` Vec<T>.  I suspect that is due to the fact that the
    //   data becomes a lot sparser.
    // * However, it seems that *eager* compaction is faster than lazy compaction.
    //   This makes sense, because on average we have less data to deal with.
    //   (24.1 vs 20.7 seconds)

    /// KLL calls this `δ`. Equivalent to a 99.9999% success rate per queried value.
    const FAILURE_PROBABILITY: f64 = 1e-6;
    /// `CAPACITY_DECAY` specifies how much smaller compactor h+1 is wrt to h.
    /// KLL calls this `c`.
    const CAPACITY_DECAY: f64 = 2.0 / 3.0;

    const MIN_COMPACTOR_SIZE: usize = 2;

    /// Smallest `k` guaranteeing rank error <= `error * n` w.p. >= 1 - `delta` for a
    /// *single* query value. Union-bound over ~`1/error` values (i.e. pass
    /// `delta * error`) if you need all quantiles to hold simultaneously.
    ///
    /// Randomized compaction makes the rank error a zero-mean sum of independent
    /// steps: compacting level `h` shifts the estimate by ±2^h, and only when the
    /// number of compacted items below the query is odd. Level `h` has capacity
    /// `k_h = k c^(H-h)` and items of weight 2^h, so it compacts at most
    /// `n / (2^h k_h)` times and, taking every step as ±2^h (worst-case parity),
    ///
    ///     Var <= sum_h 4^h * n / (2^h * k_h) = (n * 2^H / k) * sum_j (2c)^-j
    ///          = (n * 2^H / k) * 2c/(2c-1)                        [needs c > 1/2]
    ///
    /// A level above `H` only appears once the old top compactor -- capacity `k`,
    /// weight 2^(H-1) -- filled up, so `2^H <= 2n/k` and
    ///
    ///     std <= (n/k) * 2 sqrt(c / (2c - 1)).
    ///
    /// The steps are bounded, so Hoeffding gives a sub-Gaussian tail with exactly
    /// that variance proxy: the error stays below `z * std` except w.p. `delta`,
    /// with `z = sqrt(2 ln(2/delta))`. Hence
    ///
    ///     k = z * 2 sqrt(c / (2c - 1)) / error.
    ///
    /// This is the worst case; the schedule in `compact()` lets compactors run past
    /// their thresholds, so the measured std is 0.25..1.08 * n/k (k in 16..50k,
    /// n in 1e4..1e8, random/sorted/reverse-sorted input) against the 2.83 * n/k
    /// bound used here.
    fn compute_k(error: f64) -> usize {
        assert!(error > 0.0 && error < 1.0, "invalid error: {error}");

        let z = f64::sqrt(2.0 * f64::ln(2.0 / FAILURE_PROBABILITY)); // sub-Gaussian tail factor for prob. 1 - delta
        let spread = 2.0 * f64::sqrt(CAPACITY_DECAY / (2.0 * CAPACITY_DECAY - 1.0)); // std bound in units of n/k
        f64::max(MIN_COMPACTOR_SIZE as f64, f64::ceil(z * spread / error)) as usize
    }

    #[derive(Debug, Clone, Copy)]
    struct Level {
        offset: usize,
        size: usize,
        /// Number of compactions performed on this compactor.
        compactions: u64,
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
                scratch: self.scratch.clone(),
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
                levels: vec![Level {
                    offset: 0,
                    size: 0,
                    compactions: 0,
                    coin: false,
                }],
                k,
                consumed_items: 0,
                compactor_capacity: k,
                rng: SmallRng::from_rng(&mut rand::rng()),
                scratch: Vec::default(),
            };
            KLLSketch(State::Ingesting(state))
        }

        #[inline]
        pub fn update(&mut self, array: &[T]) {
            let State::Ingesting(state) = &mut self.0 else {
                invalid_state()
            };
            state.update(array);
        }

        pub fn finalize(&mut self) {
            let placeholder = State::Finalized(FinalizedState::default());
            let state = mem::replace(&mut self.0, placeholder);
            let State::Ingesting(state) = state else {
                invalid_state()
            };
            self.0 = State::Finalized(state.finalize());
        }

        pub fn estimate_rank(&self, value: &T) -> usize {
            let State::Finalized(state) = &self.0 else {
                invalid_state()
            };
            state.estimate_rank(value)
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
        pub fn update(&mut self, array: &[T]) {
            let mut offset = 0;
            while offset < array.len() {
                // Fast compare
                if self.items.len() >= self.compactor_capacity {
                    self.compact(true);
                }
                let space_left = self.compactor_capacity - self.items.len();
                debug_assert!(space_left > 0);
                let ingest_chunk_len = space_left.min(array[offset..].len());
                let ingest_items = &array[offset..offset + ingest_chunk_len];
                self.items.extend_from_slice(ingest_items);
                self.levels[0].size += ingest_items.len();
                offset += ingest_items.len();
            }
            self.consumed_items += array.len();
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
            self.levels.push(Level {
                offset: 0,
                size: 0,
                compactions: 0,
                coin: false,
            });
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
            compact_level.coin = match compact_level.compactions % 2 == 1 {
                true => !compact_level.coin,
                false => rand & 0x2 != 0,
            };
            compact_level.compactions += 1;
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
            merge_sorted(buf, next_level_items, compacted_items);
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
        usize::max(
            usize::try_from(nominal_size).expect("overflow"),
            MIN_COMPACTOR_SIZE,
        )
    }
}

pub mod req {
    use super::*;

    /// KLL calls this `δ`. Equivalent to a 99.9999% success rate per queried value.
    // const FAILURE_PROBABILITY: f64 = 1e-6;
    const FAILURE_PROBABILITY: f64 = 0.5;

    /// Stream length to parameterise a fresh sketch for.
    fn initial_n(error: f64) -> usize {
        let k = |n: usize| compute_k(error, FAILURE_PROBABILITY, n);
        let b = |n| compute_b(k(n), n);

        // Choose initial guess of n such that `error * n > 1`: at `error * n ==
        // 1` the `log2` in `compute_k` is zero and `k` overflows.
        let mut n = (f64::ceil(2.0 / error) as usize).next_power_of_two();
        // Ensure that:
        //   1. The number of protected items is not larger than the total number
        //      of items, because that would mean that no items could get promoted
        //      at all during compaction.
        //   2. The capacity of a relative compactor is larger than the maximum
        //      number of consumable items in the sketch, in that case we would
        //      not even fill up that first compactor.
        // TODO: [amber] Consider an addition factor of 8 or smth.
        while k(n) > n || b(n) >= n {
            n *= 2;
        }
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
                scratch: self.scratch.clone(),
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
    #[repr(transparent)]
    pub struct ReqSketch<T: fmt::Debug + Clone + TotalOrd>(State<T>);

    impl<T: fmt::Debug + Clone + TotalOrd> ReqSketch<T> {
        pub fn new(error: f64, hra: bool) -> Self {
            let n = initial_n(error);
            let k = compute_k(error, FAILURE_PROBABILITY, n);
            assert!(n > k, "n must be greater than k");
            let state = IngestingState {
                items: Vec::new(),
                scratch: Vec::new(),
                levels: vec![Level {
                    offset: 0,
                    size: 0,
                    compaction_schedule: 0,
                    coin: false,
                }],
                is_hra: hra,
                n,
                error,
                k,
                consumed_items: 0,
                rng: rand::make_rng(),
            };
            ReqSketch(State::Ingesting(state))
        }

        #[inline]
        pub fn num_items(&self) -> usize {
            match &self.0 {
                State::Ingesting(state) => state.consumed_items,
                State::Finalized(state) => state.num_items(),
            }
        }

        #[inline]
        pub fn update(&mut self, array: &[T]) {
            let State::Ingesting(state) = &mut self.0 else {
                invalid_state()
            };
            state.update(array);
        }

        #[inline]
        pub fn merge(&mut self, other: Self) {
            let State::Ingesting(other) = other.0 else {
                invalid_state()
            };
            let State::Ingesting(state) = &mut self.0 else {
                invalid_state()
            };
            state.merge(other);
        }

        #[inline]
        pub fn finalize(&mut self) {
            let placeholder = State::Finalized(FinalizedState::default());
            let state = mem::replace(&mut self.0, placeholder);
            let State::Ingesting(state) = state else {
                invalid_state()
            };
            self.0 = State::Finalized(state.finalize());
        }

        #[inline]
        pub fn estimate_rank(&self, value: &T) -> usize {
            let State::Finalized(state) = &self.0 else {
                invalid_state()
            };
            state.estimate_rank(value)
        }

        #[inline]
        pub fn estimate_quantile(&self, quantile: f64) -> Option<&T> {
            let State::Finalized(state) = &self.0 else {
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
        pub fn update(&mut self, array: &[T]) {
            self.lra.update(array);
            self.hra.update(array);
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

        pub fn estimate_rank(&self, value: &T) -> usize {
            let rank = self.lra.estimate_rank(value);
            match 2 * rank <= self.num_items() {
                true => rank,
                false => self.hra.estimate_rank(value),
            }
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
        pub fn update(&mut self, array: &[T]) {
            for item in array {
                self.compact_if_needed(0);
                self.items.push(item.clone());
                self.levels[0].size += 1;
                self.consumed_items += 1;
            }
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

            // TODO: [amber] Decide whether we want to add in the error factor or not.
            self.n = (self.error * self.n as f64 * self.n as f64) as usize;
            self.k = compute_k(self.error, FAILURE_PROBABILITY, self.n);
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
            self.levels.push(Level {
                offset: 0,
                size: 0,
                compaction_schedule: 0,
                coin: false,
            });
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

            let compare = |a: &T, b: &T| match self.is_hra {
                false => TotalOrd::tot_cmp(a, b),
                true => TotalOrd::tot_cmp(a, b).reverse(),
            };

            let compactor_start = self.levels[level].offset;
            let compactor_size = self.levels[level].size;
            let compactor_end = compactor_start + compactor_size;
            let compactor = &self.items[compactor_start..compactor_end];

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

            let compactor = &mut self.items[compactor_start..compactor_end];
            // Stash the protected items at the end of the compactor.
            compactor.select_nth_unstable_by(promote_count, |a, b| compare(a, b).reverse());
            // Sort the items that we will be compacting.
            compactor[..promote_count].sort_unstable_by(compare);

            // Throw away half of the values during the compaction, gathering the
            // survivors at the front of the compacted range.
            for i in 0..promote_count / 2 {
                compactor.swap(i, 2 * i + coin as usize);
            }

            // Drop the non-promoted items from the item pool.
            let gap_start = compactor_start + promote_count / 2;
            let gap_end = compactor_start + promote_count;
            self.items.drain(gap_start..gap_end);

            // Transfer ownership of the promoted items to the next compactor.
            self.levels[level + 1].size += promote_count / 2;
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

        /// Merge `other` into `self`.
        fn merge(&mut self, mut other: Self) {
            assert_eq!(self.is_hra, other.is_hra);
            assert_eq!(self.error, other.error);

            // We need a compactor for every one of `other`'s levels.
            while self.levels.len() < other.levels.len() {
                self.add_new_compactor();
            }

            let scratch = [&mut self.scratch, &mut other.scratch]
                .into_iter()
                .max_by_key(|v| v.capacity())
                .unwrap();
            mem::swap(&mut self.items, scratch);
            let items1 = scratch;
            let items2 = &mut other.items;
            self.items.clear();
            self.items.reserve_exact(items1.len() + items2.len());

            let mut next_offset = 0;
            for level in (0..self.levels.len()).rev() {
                let l1 = self.levels[level];
                let l2 = other.levels.get(level).copied().unwrap_or_default();
                let compactor1 = &items1[l1.offset..l1.offset + l1.size];
                let compactor2 = &items2[l2.offset..l2.offset + l2.size];
                self.items.extend_from_slice(compactor1);
                self.items.extend_from_slice(compactor2);

                let at_odd_schedule = |l: &Level| l.compaction_schedule % 2 != 0;
                let coin = match (at_odd_schedule(&l1), at_odd_schedule(&l2)) {
                    (false, false) => l1.coin, // Next compaction will draw a fresh coin, so we don't care.
                    (true, false) => l1.coin,
                    (false, true) => l2.coin,
                    (true, true) if l1.coin == l2.coin => l1.coin,
                    (true, true) => self.rng.random(),
                };

                self.levels[level] = Level {
                    offset: next_offset,
                    size: l1.size + l2.size,
                    compaction_schedule: l1.compaction_schedule | l2.compaction_schedule,
                    coin,
                };
                next_offset += l1.size + l2.size;
            }
            debug_assert_eq!(next_offset, self.items.len());

            self.n = usize::max(self.n, other.n);
            self.k = compute_k(self.error, FAILURE_PROBABILITY, self.n);
            self.consumed_items += other.consumed_items;

            for level in 0..self.levels.len() {
                self.compact_if_needed(level);
            }
        }

        fn finalize(self) -> FinalizedState<T> {
            let IngestingState {
                mut items,
                levels,
                consumed_items,
                ..
            } = self;

            let pool_size: usize = items.len();
            dbg!(&pool_size);

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
            let mut finalized_items = Vec::with_capacity(items.len());
            let cum_weights = finalize_merge_levels(&level_items, &mut finalized_items);

            debug_assert_eq!(cum_weights.last().unwrap_or(&0), &consumed_items);

            FinalizedState::new(
                finalized_items.into_boxed_slice(),
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

fn merge_sorted<T: TotalOrd>(
    vec: &mut Vec<T>,
    iter1: impl ExactSizeIterator<Item = T>,
    iter2: impl ExactSizeIterator<Item = T>,
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
                if TotalOrd::tot_le(x1, x2) {
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
    pub fn update(&mut self, array: &[T]) {
        match self {
            Sketch::Kll(s) => s.update(array),
            Sketch::Req(s) => s.update(array),
            Sketch::DoubleReq(s) => s.update(array),
        }
    }

    pub fn merge(&mut self, other: Self) {
        match (self, other) {
            // TODO: [amber] KLLSketch has no merge yet.
            (Sketch::Kll(_), Sketch::Kll(_)) => todo!(),
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

    pub fn estimate_rank(&self, value: &T) -> usize {
        match self {
            Sketch::Kll(s) => s.estimate_rank(value),
            Sketch::Req(s) => s.estimate_rank(value),
            Sketch::DoubleReq(s) => s.estimate_rank(value),
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
    use super::ApproxQuantileMethod;
    use super::kll::KLLSketch;
    use super::req::ReqSketch;

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
                        base.update(&data[..5_000]);
                        let (mut a, mut b) = (base.clone(), base.clone());
                        a.update(&data[5_000..]);
                        b.update(&data[5_000..]);
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
