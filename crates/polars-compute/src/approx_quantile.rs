use std::ops::RangeInclusive;
use std::str::FromStr;
use std::{fmt, mem, slice};

pub use kll::KLLSketch;
use polars_error::{PolarsError, PolarsResult, polars_bail, polars_ensure};
use polars_utils::itertools::Itertools;
use polars_utils::total_ord::TotalOrd;
use rand::RngExt;
use rand::rngs::SmallRng;
pub use req::{DoubleReqSketch, ReqSketch};

/// Compute these quantile values using KLL, if the method is Auto.
const KLL_RANGE: RangeInclusive<f64> = 0.05..=0.95;

/// The probability that a query exceeds `error`. KLL calls this `δ`.
///
/// Taken from the 3-sigma rule.
const FAILURE_PROBABILITY: f64 = 1.0 - 0.9973;

/// Smallest error a sketch can be parameterised for.
pub const MIN_ERROR: f64 = 1.0 / (1u64 << 32) as f64;

/// Looseness of the formal KLL error bound (estimated by measuring).
const KLL_BOUND_LOOSENESS: f64 = 6.0;
/// Looseness of the formal REQ error bound (estimated by measuring).
const REQ_BOUND_LOOSENESS: f64 = 20.0;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ApproxQuantileMethod {
    Auto,
    KLL,
    ReqSketch { hra: bool },
    DoubleReqSketch,
}

impl FromStr for ApproxQuantileMethod {
    type Err = PolarsError;

    fn from_str(s: &str) -> PolarsResult<Self> {
        Ok(match s {
            "auto" => Self::Auto,
            "kll" => Self::KLL,
            "req_lo" => Self::ReqSketch { hra: false },
            "req_hi" => Self::ReqSketch { hra: true },
            "req_both" => Self::DoubleReqSketch,
            v => {
                polars_bail!(
                    InvalidOperation:
                    "`method` must be one of {{'auto', 'kll', 'req_lo', 'req_hi', 'req_both'}}, got {v}"
                )
            },
        })
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

    /// Translate an empirically calibrated error into the formal bound of this method.
    pub fn empirical_error_to_formal(&self, empirical_error: f64) -> f64 {
        // The maths get weird at errors close to 1 (especially for REQ), so cap it below that.
        const MAX_ERROR: f64 = 0.90;
        match self {
            Self::Auto => panic!("method not resolved"),
            Self::KLL => f64::min(MAX_ERROR, KLL_BOUND_LOOSENESS * empirical_error),
            Self::ReqSketch { .. } | Self::DoubleReqSketch => {
                f64::min(MAX_ERROR, REQ_BOUND_LOOSENESS * empirical_error)
            },
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FinalizedSketch<T: fmt::Debug + Clone + TotalOrd> {
    /// All retained items, sorted.
    items: Box<[T]>,
    /// Inclusive cumulative weight, i.e. `cum_weight[i]` is the 1-based rank of
    /// `items[i]`. `None` when every item has weight 1.
    cum_weight: Option<Box<[u64]>>,
    /// Number of times the weights count every ingested item.
    weight_factor: u64,
}

impl<T: fmt::Debug + Clone + TotalOrd> FinalizedSketch<T> {
    fn new(items: Box<[T]>, cum_weight: Option<Box<[u64]>>) -> Self {
        Self {
            items,
            cum_weight,
            weight_factor: 1,
        }
    }

    /// The number of items this sketch ingested.
    fn num_items(&self) -> u64 {
        debug_assert!(self.total_weight().is_multiple_of(self.weight_factor));
        self.total_weight() / self.weight_factor
    }

    /// The summed weight of all retained items.
    fn total_weight(&self) -> u64 {
        match &self.cum_weight {
            Some(cum_weight) => cum_weight.last().copied().unwrap_or(0),
            None => self.items.len() as u64,
        }
    }

    pub fn estimate_quantile(&self, quantile: f64) -> PolarsResult<Option<&T>> {
        polars_ensure!(
            (0.0..=1.0).contains(&quantile),
            ComputeError: "`quantile` should be between 0.0 and 1.0",
        );
        // We round with ties toward ∞ for consistency with the regular quantile.
        let num_items = self.num_items();
        let item_rank = (quantile * num_items.saturating_sub(1) as f64).round();
        let weighted_rank = match num_items {
            0 => return Ok(None),
            1 => 1,
            _ => {
                1 + (item_rank * (self.total_weight() - 1) as f64 / (num_items - 1) as f64).round()
                    as u64
            },
        };
        let idx = estimate_quantile_index(self.cum_weight.as_deref(), weighted_rank);
        Ok(Some(&self.items[idx]))
    }

    /// Split into the retained items and each item's individual weight.
    fn into_items_and_weights(self) -> (Vec<T>, Vec<u64>) {
        let items = self.items.into_vec();
        let weights = match self.cum_weight {
            Some(cum_weight) => {
                let mut weights = cum_weight.into_vec();
                let mut last = 0;
                for w in weights.iter_mut() {
                    (*w, last) = (*w - last, *w);
                }
                weights
            },
            None => vec![1; items.len()],
        };
        (items, weights)
    }

    /// Combine two sketches of the same items into one, summing their weights.
    fn merge_halves(s1: Self, s2: Self) -> Self {
        assert_eq!(s1.num_items(), s2.num_items());
        let weight_factor = s1.weight_factor + s2.weight_factor;

        if s1.cum_weight.is_none() && s2.cum_weight.is_none() {
            let mut items = Vec::new();
            merge_sorted(
                &mut items,
                s1.items.into_vec().into_iter(),
                s2.items.into_vec().into_iter(),
                TotalOrd::tot_cmp,
            );
            return Self {
                items: items.into_boxed_slice(),
                cum_weight: None,
                weight_factor,
            };
        }

        let (items1, weights1) = s1.into_items_and_weights();
        let (items2, weights2) = s2.into_items_and_weights();

        let mut merged = Vec::new();
        merge_sorted(
            &mut merged,
            Itertools::zip_eq(items1.into_iter(), weights1),
            Itertools::zip_eq(items2.into_iter(), weights2),
            |(item1, _), (item2, _)| TotalOrd::tot_cmp(item1, item2),
        );

        let mut items = Vec::with_capacity(merged.len());
        let mut cum_weights = Vec::with_capacity(merged.len());
        let mut cum_weight = 0;
        for (item, weight) in merged {
            cum_weight += weight;
            items.push(item);
            cum_weights.push(cum_weight);
        }

        Self {
            items: items.into_boxed_slice(),
            cum_weight: Some(cum_weights.into_boxed_slice()),
            weight_factor,
        }
    }
}

#[inline(never)]
fn estimate_quantile_index(cum_weight: Option<&[u64]>, estimated_rank: u64) -> usize {
    match cum_weight {
        Some(cum_weight) => cum_weight.partition_point(|w| *w < estimated_rank),
        None => estimated_rank as usize - 1,
    }
}

pub mod kll {
    use super::*;

    /// `CAPACITY_DECAY` specifies how much smaller compactor h+1 is wrt to h.
    /// KLL calls this `c`.
    const CAPACITY_DECAY: f64 = 2.0 / 3.0;

    /// Require compactors to be at least this size.
    const MIN_COMPACTOR_SIZE: usize = 2;

    /// Compactors at or below this capacity are replaced by [`Sampler`].
    const SAMPLER_CUTOFF: usize = 8;

    /// Smallest `k` guaranteeing rank error <= `error * n` w.p. >= 1 - `delta` for a
    /// *single* query value, with `delta` = `FAILURE_PROBABILITY`.
    #[inline(never)]
    fn compute_k(error: f64) -> usize {
        assert!((MIN_ERROR..1.0).contains(&error), "invalid error: {error}");

        // `Σ_{d >= 1} r^d` for `r < 1`.
        let geometric_tail = |r| r / (1.0 - r);

        // Hoeffding's tail `2 exp(-t² / 2Σw²)` (KLL Lemma 1) at `t = εn`, solved for `Σw²`.
        let z = f64::sqrt(2.0 * f64::ln(2.0 / FAILURE_PROBABILITY));

        // Compute k from the total variance (`Σw²`) and multiply by the "error spread" (`•/error`)
        let k_from_total_variance = |var: f64| z * f64::sqrt(var) / error;

        // `Σw²` of the compactors in units of (n/k)²:
        //   * Where c is the CAPACITY_DECAY.
        //   * Where h is the height of a compactor (counting from 0).
        //   * Let d(h) := H - 1 - h (the depth of a compactor).
        //   * c^d is the size of a compactor at depth d.
        //   * 2^h is the number of items that get inserted into the compactor).
        //   * Each compactor leads to at most `y_h := n / (k * c^d(h) * 2^h)` "weight shifts".
        //   * So each compactor weight-shifts y_(h+1) / y_h times the one below it:
        //     == (n/(k * c^d(h+1) * 2^(h+1))) / (n/(k * c^d(h) * 2^h))
        //     == (c^d(h) / c^d(h+1)) * (2^h / 2^(h+1))
        //     == c^(H - 1 - h - H + 1 + (h+1)) * 2^(h - h + 1)
        //     == 2*c
        //  * So the the total number of all weight shifts is equal to:
        //    (1/(2c) + 1/(2c)^2 + 1/(2c)^3 + ...)
        //  * We know that the creation of level H-1 consumed k items of weight 2^(H-2).
        //    The sum of the weights of those items denote that at least that many
        //    items were ingested in total, i.e., `k * 2^(H-2) ≤ n.`
        //    Rewrite ⇒ n/k ≥ 2^(H-2) ⇒ 2^(H-1) ≤ 2*(n/k).
        //  * Finish: Total variance (in terms of n/k) is 2 * ((1/(2c) + 1/(2c)^2 + ...).
        let compactor_var = 2.0 * geometric_tail(1.0 / (2.0 * CAPACITY_DECAY));

        // `Σw²` of the sampler in units of (n/k)²:
        //   * `feed` conserves weight, so at most `n / 2^h` emissions happen at height `h`,
        //     each a step of size `2^h` (KLL Lemma 2). Summed, that is at most `n * 2^L`.
        //   * Level L-1 sits at depth D = H - L and was replaced, so `k * c^D ≤ CUTOFF`.
        //     Hence `2^D ≥ (k / CUTOFF)^α` where `α = ln 2 / ln(1/c)` (KLL Theorem. 2).
        //   * Recall `n/k ≥ 2^(H-2)` ⇒ `2^H ≤ 4 * n/k`.
        //   * Compute the full variance:
        //       n * 2^L = n * 2^H / 2^D ≤ (4 * n²/k) / (k / CUTOFF)^α
        //               = (n²/k²) * 4 * k * (CUTOFF / k)^α
        let alpha = f64::ln(2.0) / f64::ln(1.0 / CAPACITY_DECAY);
        let sampler_var = |k: f64| 4.0 * k * f64::powf(SAMPLER_CUTOFF as f64 / k, alpha);

        let k0 = k_from_total_variance(compactor_var);
        let k1 = k_from_total_variance(compactor_var + sampler_var(k0));
        let k2 = k_from_total_variance(compactor_var + sampler_var(k1));
        debug_assert!(k0 <= k1 && k1 <= k2, "k does not converge downward");
        usize::max(MIN_COMPACTOR_SIZE, k2.ceil() as usize)
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

    /// Stand-in for the run of [`SAMPLER_CUTOFF`]-sized compactors at the bottom
    /// of the chain, which is a uniform sampler in distribution.
    #[derive(Debug, Clone)]
    struct Sampler<T> {
        /// The retained item. `Some` whenever `weight` is nonzero.
        item: Option<T>,
        /// Weight `item` stands for, below the weight at which it is promoted.
        weight: u64,
    }

    impl<T> Sampler<T> {
        /// Absorb `item` of weight `item_weight` into the sample.
        ///
        /// Once per window of `promote_weight` size, return a uniformly sampled
        /// item from the window.
        #[inline]
        fn feed(
            &mut self,
            item_weight: u64,
            promote_weight: u64,
            rng: &mut SmallRng,
            to_owned_item: impl FnOnce() -> T,
        ) -> Option<T> {
            debug_assert!(0 < item_weight && item_weight < promote_weight);
            debug_assert_eq!(self.weight == 0, self.item.is_none());

            let total_weight = self.weight + item_weight;
            if total_weight < promote_weight {
                // Probabilistically replace the internal value.
                if rng.random_range(0..total_weight) < item_weight {
                    self.item = Some(to_owned_item());
                }
                self.weight = total_weight;
                None
            } else {
                // Weight overflow: randomly emit either of the two compactor values.
                let carry = total_weight - promote_weight;
                let emit_internal_value =
                    rng.random_range(0..promote_weight - carry) < self.weight - carry;
                let held = self.item.take();
                self.weight = carry;
                let (emitted, kept) = match emit_internal_value {
                    true => (held, (carry > 0).then(to_owned_item)),
                    false => (Some(to_owned_item()), held.filter(|_| carry > 0)),
                };
                self.item = kept;
                emitted
            }
        }
    }

    #[derive(Debug)]
    struct IngestingState<T: fmt::Debug + Clone + TotalOrd> {
        /// Contents of the compactors.
        ///
        /// This algorithm uses the convention that the top-level compactor has
        /// *level* h-1.  The bottom-level compactor has *level* h,
        /// and height *0*. So the order of `levels` is *reversed* wrt `items`.
        items: Vec<T>,
        levels: Vec<Level>,
        k: usize,
        /// Total number of items that were consumed by this sketch.
        consumed_items: u64,
        /// Maximum number of items before we compact.
        total_capacity: usize,
        sampler: Sampler<T>,
        sampler_level: usize,
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
                total_capacity: self.total_capacity,
                sampler: self.sampler.clone(),
                sampler_level: self.sampler_level,
                rng: rand::make_rng(),
                scratch: Vec::new(),
            }
        }
    }

    #[derive(Debug, Clone)]
    #[repr(transparent)]
    pub struct KLLSketch<T: fmt::Debug + Clone + TotalOrd>(IngestingState<T>);

    impl<T: fmt::Debug + Clone + TotalOrd> KLLSketch<T> {
        pub fn new(error: f64) -> Self {
            let k = compute_k(error);
            let state = IngestingState {
                items: Vec::new(),
                levels: vec![Level::default()],
                k,
                consumed_items: 0,
                total_capacity: k,
                rng: rand::make_rng(),
                scratch: Vec::default(),
                sampler_level: 0,
                sampler: Sampler {
                    item: None,
                    weight: 0,
                },
            };
            KLLSketch(state)
        }

        #[inline]
        pub fn update<Q: ToOwned<Owned = T> + ?Sized>(&mut self, item: &Q) {
            self.0.update(item);
        }

        pub fn merge(&mut self, other: &Self) {
            self.0.merge(&other.0);
        }

        /// Stop ingesting, keeping only what this sketch retained.
        pub fn finalize(self) -> FinalizedSketch<T> {
            self.0.finalize()
        }
    }

    impl<T: fmt::Debug + Clone + TotalOrd> IngestingState<T> {
        #[inline]
        pub fn update<Q: ToOwned<Owned = T> + ?Sized>(&mut self, item: &Q) {
            self.consumed_items += 1;
            if self.items.len() >= self.total_capacity {
                self.compact(true);
            }
            let level = self.sampler_level;
            let sampled = if self.sampler_is_activated() {
                let Self { sampler, rng, .. } = self;
                sampler.feed(1, 1 << level, rng, || item.to_owned())
            } else {
                Some(item.to_owned())
            };
            if let Some(item) = sampled {
                self.update_compactors(item);
            }
        }

        fn sampler_is_activated(&self) -> bool {
            self.sampler_level > 0
        }

        fn update_compactors(&mut self, item: T) {
            self.items.push(item);
            self.levels[self.sampler_level].size += 1;
            for empty_compactor in self.levels[..self.sampler_level].iter_mut() {
                debug_assert_eq!(empty_compactor.size, 0, "compactor should be empty");
                empty_compactor.offset = self.items.len();
            }
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
                        return;
                    };
                }
            }
        }

        fn add_new_compactor(&mut self) {
            self.levels.push(Level::default());
            self.recompute_sampler_level();
            self.flush_to_sampler();
            self.recompute_total_capacity();
        }

        /// The lowest level whose compactor is larger than [`SAMPLER_CUTOFF`].
        fn recompute_sampler_level(&mut self) {
            let sampler_level = (0..self.levels.len())
                .filter(|level| {
                    compactor_threshold(self.k, self.levels.len() - 1 - level) <= SAMPLER_CUTOFF
                })
                .count();
            // Make sure there is space for the sampler output to go.
            let max_sampler_level = self.levels.len().saturating_sub(2);
            self.sampler_level = usize::min(sampler_level, max_sampler_level);
        }

        /// Move every item below `sampler_level` into the sampler.
        fn flush_to_sampler(&mut self) {
            let sampler_level = self.sampler_level;
            if sampler_level == 0 {
                return;
            }
            let base_compactor = self.levels[sampler_level];
            let start = base_compactor.offset + base_compactor.size;
            let promote_weight = 1u64 << sampler_level;

            debug_assert!(self.scratch.is_empty());
            let mut promoted = mem::take(&mut self.scratch);

            let mut drain = self.items.drain(start..);
            for level in (0..sampler_level).rev() {
                let compactor = &mut self.levels[level];
                for item in drain.by_ref().take(compactor.size) {
                    promoted.extend(self.sampler.feed(
                        1u64 << level,
                        promote_weight,
                        &mut self.rng,
                        || item,
                    ));
                }
                compactor.offset = start;
                compactor.size = 0;
            }
            debug_assert_eq!(drain.len(), 0);
            drop(drain);

            for item in promoted.drain(..) {
                self.update_compactors(item);
            }
            let _ = mem::replace(&mut self.scratch, promoted);
        }

        fn recompute_total_capacity(&mut self) {
            self.total_capacity = (0..self.levels.len() - self.sampler_level)
                .map(|depth| compactor_threshold(self.k, depth))
                .sum::<usize>()
                .next_multiple_of(2);
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
            debug_assert!(self.scratch.is_empty());
            let buf = &mut self.scratch;

            // If there is an odd number of items in this compactor, stash the "straggler" to add it back later
            let mut straggler = None;
            if !compact_level.size.is_multiple_of(2) {
                if coin1 {
                    straggler = Some(self.items[compact_start].clone());
                    compact_start += 1;
                } else {
                    straggler = Some(self.items[old_compact_end - 1].clone());
                    compact_end -= 1;
                }
            }

            // The base compactor is not sorted yet
            if level <= self.sampler_level {
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
            self.items[next_start..next_start + buf.len()].clone_from_slice(buf);
            next_level.size = buf.len();
            buf.clear();

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

        /// Merge `other` into `self`.
        fn merge(&mut self, other: &Self) {
            // `k` is a function of the error, so k₁ = k₂ ⇒ ε₁ = ε₂.
            assert_eq!(self.k, other.k);

            // Make sure we have enough compactors on the left side.
            while self.levels.len() < other.levels.len() {
                self.add_new_compactor();
            }
            debug_assert!(self.sampler_level >= other.sampler_level);

            let mut items = Vec::with_capacity(self.items.len() + other.items.len());
            let items1 = mem::take(&mut self.items);
            let items2 = &other.items;

            let mut next_offset = 0;
            for level in (0..self.levels.len()).rev() {
                let l1 = self.levels[level];
                let l2 = other.levels.get(level).copied().unwrap_or_default();
                let comp1 = &items1[l1.offset..l1.offset + l1.size];
                let comp2 = &items2[l2.offset..l2.offset + l2.size];
                if level == self.sampler_level {
                    items.extend_from_slice(comp1);
                    items.extend_from_slice(comp2);
                } else {
                    let (c1, c2) = (comp1.iter().cloned(), comp2.iter().cloned());
                    merge_sorted(&mut items, c1, c2, TotalOrd::tot_cmp);
                }

                self.levels[level] = Level {
                    offset: next_offset,
                    size: l1.size + l2.size,
                    mid_pair: l1.mid_pair | l2.mid_pair,
                    coin: merge_coin(l1.mid_pair, l1.coin, l2.mid_pair, l2.coin, &mut self.rng),
                };
                next_offset += l1.size + l2.size;
            }
            debug_assert_eq!(next_offset, items.len());
            self.items = items;

            self.consumed_items += other.consumed_items;
            // Absorb the low-weight compactors into our sampler.
            self.flush_to_sampler();
            if let Some(item) = other.sampler.item.as_ref() {
                let level = self.sampler_level;
                let Self { sampler, rng, .. } = self;
                let promoted = sampler.feed(other.sampler.weight, 1 << level, rng, || item.clone());
                if let Some(item) = promoted {
                    self.update_compactors(item);
                }
            }
            self.compact(false);
        }

        fn finalize(self) -> FinalizedSketch<T> {
            let IngestingState {
                mut items,
                levels,
                consumed_items,
                mut scratch,
                sampler_level,
                sampler,
                ..
            } = self;

            // Base level is not yet sorted
            let base = levels[sampler_level];
            items[base.offset..base.offset + base.size].sort_unstable_by(TotalOrd::tot_cmp);

            // With a single compactor every item has weight 1.
            if levels.len() == 1 {
                debug_assert_eq!(sampler.weight, 0);
                return FinalizedSketch::new(items.into_boxed_slice(), None);
            }

            // Merge all sorted levels
            let mut level_items: Vec<&[T]> = levels
                .iter()
                .map(|level| &items[level.offset..level.offset + level.size])
                .collect();
            let num_levels = levels.len();
            if let Some(item) = &sampler.item {
                level_items.push(slice::from_ref(item));
            }
            let level_to_weight = |level| {
                if level == num_levels {
                    sampler.weight
                } else {
                    1u64 << level
                }
            };
            let cum_weights = finalize_merge_levels(&level_items, level_to_weight, &mut scratch);

            debug_assert_eq!(
                scratch.len(),
                items.len() + usize::from(sampler.item.is_some())
            );
            debug_assert_eq!(cum_weights.last().unwrap_or(&0), &consumed_items);

            FinalizedSketch::new(
                scratch.into_boxed_slice(),
                Some(cum_weights.into_boxed_slice()),
            )
        }
    }

    /// Capacity of the compactor `depth` levels below the top: `ceil(k (2/3)^depth)`
    /// rounded up to an even number, and never below `MIN_COMPACTOR_SIZE`.
    fn compactor_threshold(k: usize, depth: usize) -> usize {
        let nominal_size =
            ((k as u128) * 2u128.pow(depth as u32)).div_ceil(3u128.pow(depth as u32));
        let nominal_size = usize::try_from(nominal_size).expect("overflow");
        usize::max(nominal_size.next_multiple_of(2), MIN_COMPACTOR_SIZE)
    }
}

pub mod req {
    use super::*;

    /// Stream length to parameterise a fresh sketch for.
    fn initial_n(error: f64) -> usize {
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
        let n_is_ok = |n| {
            let k = compute_k(error, n);
            n > k && n > compute_b(k, n)
        };
        while !n_is_ok(n) {
            n = n.checked_mul(2).expect("no sketch size fits this error");
        }
        n
    }

    fn compute_k(error: f64, n: usize) -> usize {
        assert!((MIN_ERROR..1.0).contains(&error), "invalid error: {error}");

        // Eq. 6
        let k = 2 * f64::ceil(
            (4.0 / error)
                * f64::sqrt((-f64::ln(FAILURE_PROBABILITY)) / f64::log2(error * n as f64)),
        ) as usize;
        assert!(k >= 2);
        k
    }

    fn compute_b(k: usize, n: usize) -> usize {
        // Sec 2.1: k is an *even* integer parameter.
        debug_assert!(
            k > 0 && k.is_multiple_of(2),
            "k must be a positive even integer"
        );
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
        /// for. Doubled on every growth.
        n: usize,
        /// The allowed error as a fraction of `n`.
        error: f64,
        /// k parameter of the paper: the size of a compactor section. Impacts
        /// how many items are protected during a compaction. Shrinks over time,
        /// see `close_out_if_needed`.
        k: usize,
        consumed_items: u64,
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
    #[repr(transparent)]
    pub struct ReqSketch<T: fmt::Debug + Clone + TotalOrd>(IngestingState<T>);

    impl<T: fmt::Debug + Clone + TotalOrd> ReqSketch<T> {
        pub fn new(error: f64, hra: bool) -> Self {
            let n = initial_n(error);
            let k = compute_k(error, n);
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
            ReqSketch(state)
        }

        #[inline]
        pub fn update<Q: ToOwned<Owned = T> + ?Sized>(&mut self, item: &Q) {
            self.0.update(item);
        }

        pub fn merge(&mut self, other: &Self) {
            assert_eq!(self.0.is_hra, other.0.is_hra);
            self.0.merge(&other.0);
        }

        /// Stop ingesting, keeping only what this sketch retained.
        pub fn finalize(self) -> FinalizedSketch<T> {
            self.0.finalize()
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
        pub fn update<Q: ToOwned<Owned = T> + ?Sized>(&mut self, item: &Q) {
            self.lra.update(item);
            self.hra.update(item);
        }

        pub fn merge(&mut self, other: &Self) {
            self.lra.merge(&other.lra);
            self.hra.merge(&other.hra);
        }

        /// Stop ingesting, keeping only what both sketches retained.
        pub fn finalize(self) -> FinalizedSketch<T> {
            FinalizedSketch::merge_halves(self.lra.finalize(), self.hra.finalize())
        }
    }

    impl<T: fmt::Debug + Clone + TotalOrd> IngestingState<T> {
        #[inline]
        pub fn update<Q: ToOwned<Owned = T> + ?Sized>(&mut self, item: &Q) {
            self.compact_if_needed(0);
            self.items.push(item.to_owned());
            self.levels[0].size += 1;
            self.consumed_items += 1;
        }

        /// Grow the compactors once a compaction schedule runs out of sections.
        fn close_out_if_needed(&mut self, level: usize) {
            if self.num_sections() >= 64 {
                // We assume that the compaction schedule will never overflow over 64 bits.
                return;
            }

            let schedule = self.levels[level].compaction_schedule;
            let sections_needed = u64::BITS - schedule.leading_zeros();

            // The paper squares here, but growing is quite cheap in practice,
            // so we just amortize by doubling.
            while sections_needed >= self.num_sections() as u32 {
                self.n = self.n.checked_mul(2).expect("overflow");
                self.k = compute_k(self.error, self.n);
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

        /// Merge `other` into `self`.
        fn merge(&mut self, other: &Self) {
            assert_eq!(self.is_hra, other.is_hra);
            assert_eq!(self.error, other.error);

            // We need a compactor for every one of `other`'s levels.
            while self.levels.len() < other.levels.len() {
                self.add_new_compactor();
            }

            let mut items = Vec::with_capacity(self.items.len() + other.items.len());
            let items1 = mem::take(&mut self.items);
            let items2 = &other.items;

            let mut next_offset = 0;
            for level in (0..self.levels.len()).rev() {
                let l1 = self.levels[level];
                let l2 = other.levels.get(level).copied().unwrap_or_default();
                let comp1 = &items1[l1.offset..l1.offset + l1.size];
                let comp2 = &items2[l2.offset..l2.offset + l2.size];
                if level == 0 {
                    items.extend_from_slice(comp1);
                    items.extend_from_slice(comp2);
                } else {
                    let (c1, c2) = (comp1.iter().cloned(), comp2.iter().cloned());
                    match self.is_hra {
                        false => merge_sorted(&mut items, c1, c2, cmp_desc::<false, T>),
                        true => merge_sorted(&mut items, c1, c2, cmp_desc::<true, T>),
                    }
                }

                let mid_pair = |l: &Level| !l.compaction_schedule.is_multiple_of(2);

                self.levels[level] = Level {
                    offset: next_offset,
                    size: l1.size + l2.size,
                    compaction_schedule: l1.compaction_schedule | l2.compaction_schedule,
                    coin: merge_coin(
                        mid_pair(&l1),
                        l1.coin,
                        mid_pair(&l2),
                        l2.coin,
                        &mut self.rng,
                    ),
                };
                next_offset += l1.size + l2.size;
            }
            debug_assert_eq!(next_offset, items.len());
            self.items = items;

            self.n = usize::max(self.n, other.n);
            self.k = compute_k(self.error, self.n);
            self.consumed_items += other.consumed_items;

            for level in 0..self.levels.len() {
                self.compact_if_needed(level);
            }
        }

        /// Compact `level` if it is full.
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
            let capacity = self.compactor_capacity();
            debug_assert!(self.levels[level].size >= capacity, "compactor is not full");
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
            debug_assert!(l_c.is_multiple_of(2));
            debug_assert!(promote_count >= l_c);

            // Only draw a fresh promotion parity every other compaction, and take
            // the opposite one in between. See DOI 10.3390/s22249612, Sec 3.2.
            let coin = match !self.levels[level].compaction_schedule.is_multiple_of(2) {
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
                debug_assert!(self.scratch.is_empty());
                match self.is_hra {
                    false => merge_sorted(&mut self.scratch, left, right, cmp_desc::<false, T>),
                    true => merge_sorted(&mut self.scratch, left, right, cmp_desc::<true, T>),
                }
                self.items[next].clone_from_slice(&self.scratch);
                self.scratch.clear();
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

        fn finalize(self) -> FinalizedSketch<T> {
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
                return FinalizedSketch::new(items.into_boxed_slice(), None);
            }

            // Merge all sorted levels
            let level_items: Vec<&[T]> = levels
                .iter()
                .map(|level| &items[level.offset..level.offset + level.size])
                .collect();
            let cum_weights = finalize_merge_levels(&level_items, |level| 1 << level, &mut scratch);

            debug_assert_eq!(scratch.len(), items.len());
            debug_assert_eq!(cum_weights.last().unwrap_or(&0), &consumed_items);

            FinalizedSketch::new(
                scratch.into_boxed_slice(),
                Some(cum_weights.into_boxed_slice()),
            )
        }
    }
}

/// H-way merge-sort of the per-level sorted runs into a single sorted run, and
/// the inclusive cumulative weight of every merged item, weighing the items of
/// `levels[i]` by `weight(i)`.
fn finalize_merge_levels<T: fmt::Debug + Clone + TotalOrd>(
    levels: &[&[T]],
    weight: impl Fn(usize) -> u64,
    out: &mut Vec<T>,
) -> Vec<u64> {
    let num_items: usize = levels.iter().map(|level| level.len()).sum();
    debug_assert!(out.is_empty());
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
        let cum_weight = cum_weights.last().unwrap_or(&0) + weight(level_idx);
        out.push(next_value(level_idx, &cursors).clone());
        cum_weights.push(cum_weight);
        cursors[level_idx] += 1;
    }

    debug_assert_eq!(out.len(), num_items);
    debug_assert_eq!(cum_weights.len(), num_items);
    cum_weights
}

fn merge_coin(mid1: bool, coin1: bool, mid2: bool, coin2: bool, rng: &mut SmallRng) -> bool {
    match (mid1, mid2) {
        (true, true) if coin1 != coin2 => rng.random(),
        (false, true) => coin2,
        _ => coin1,
    }
}

/// Append the merge of two runs, both sorted by `compare`, to `vec`.
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
    pub fn update<Q: ToOwned<Owned = T> + ?Sized>(&mut self, item: &Q) {
        match self {
            Sketch::Kll(s) => s.update(item),
            Sketch::Req(s) => s.update(item),
            Sketch::DoubleReq(s) => s.update(item),
        }
    }

    pub fn merge(&mut self, other: &Self) {
        match (self, other) {
            (Sketch::Kll(a), Sketch::Kll(b)) => a.merge(b),
            (Sketch::Req(a), Sketch::Req(b)) => a.merge(b),
            (Sketch::DoubleReq(a), Sketch::DoubleReq(b)) => a.merge(b),
            _ => panic!("cannot merge sketches of a different method"),
        }
    }

    pub fn finalize(self) -> FinalizedSketch<T> {
        match self {
            Sketch::Kll(s) => s.finalize(),
            Sketch::Req(s) => s.finalize(),
            Sketch::DoubleReq(s) => s.finalize(),
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
                        for v in &data[..5_000] {
                            base.update(v);
                        }
                        let (mut a, mut b) = (base.clone(), base.clone());
                        for v in &data[5_000..] {
                            a.update(v);
                            b.update(v);
                        }
                        let (a, b) = (a.finalize(), b.finalize());
                        QUANTILES.iter().all(|q| {
                            a.estimate_quantile(*q).unwrap() == b.estimate_quantile(*q).unwrap()
                        })
                    })
                    .count();
                assert!(agreed <= 2, "{} clones agreed {agreed}/10 times", $name);
            }};
        }

        assert_diverges!("ReqSketch", ReqSketch::new(0.01, true));
        assert_diverges!("KLLSketch", KLLSketch::new(0.01));
    }
}
