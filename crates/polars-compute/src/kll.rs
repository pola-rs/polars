use std::{fmt, mem};

use polars_utils::total_ord::TotalOrd;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

// TODO: [amber] Reseed on clone()

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
}

#[derive(Debug, Clone)]
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
    max_size: usize,
    rng: SmallRng,
    scratch: Vec<T>,
}

#[derive(Debug, Clone)]
pub(super) struct FinalizedState<T: fmt::Debug + Clone + TotalOrd> {
    items: Box<[T]>,
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
            levels: vec![Level { offset: 0, size: 0 }],
            k,
            consumed_items: 0,
            max_size: k,
            rng: SmallRng::from_rng(&mut rand::rng()),
            scratch: Vec::default(),
        };
        KLLSketch(State::Ingesting(state))
    }

    #[inline]
    pub fn update(&mut self, array: &[T]) {
        let State::Ingesting(state) = &mut self.0 else {
            unreachable!()
        };
        state.update(array);
    }

    pub fn finalize(&mut self) {
        let placeholder = State::Finalized(FinalizedState {
            items: Box::new([]),
            cum_weight: None,
        });
        let state = mem::replace(&mut self.0, placeholder);
        let State::Ingesting(state) = state else {
            unreachable!()
        };
        self.0 = State::Finalized(state.finalize());
    }

    pub fn estimate_rank(&self, value: &T) -> usize {
        let State::Finalized(state) = &self.0 else {
            unreachable!()
        };
        state.estimate_rank(value)
    }

    pub fn estimate_quantile(&self, quantile: f64) -> &T {
        let State::Finalized(state) = &self.0 else {
            unreachable!()
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
            if self.items.len() >= self.max_size {
                self.compact(true);
            }
            let space_left = self.max_size - self.items.len();
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
            if self.levels[level].size >= compactor_threshold(self.k, self.levels.len() - 1 - level)
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
        self.levels.push(Level { offset: 0, size: 0 });
        self.max_size = (0..self.levels.len())
            .map(|level| compactor_threshold(self.k, self.levels.len() - 1 - level))
            .sum();
    }

    fn compact_level(&mut self, level: usize) {
        let rand: u8 = self.rng.random();
        let coin1 = rand & 0x1 != 0;
        let coin2 = rand & 0x2 != 0;

        let mut compact_level = self.levels[level];
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

        // Check that all the offsets are correct
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
            mut scratch,
            ..
        } = self;

        // Base level is not yet sorted
        let base = levels[0];
        items[base.offset..base.offset + base.size].sort_unstable_by(TotalOrd::tot_cmp);

        if levels.len() == 1 {
            return FinalizedState {
                items: items.into_boxed_slice(),
                cum_weight: None,
            };
        }

        // Merge all sorted levels
        scratch.clear();
        let mut finalized_items = scratch;
        let mut cum_weights = Vec::with_capacity(items.len());
        let mut cursors: Vec<usize> = vec![0; levels.len()];

        // Are we done draining this level?
        let is_done = |level: usize, cursors: &[usize]| cursors[level] >= levels[level].size;
        // Get the next value corresponding to level `level`.
        let next_value =
            |level: usize, cursors: &[usize]| &items[levels[level].offset + cursors[level]];

        // H-way merge-sort
        finalized_items.reserve_exact(items.len());
        while let Some(level_idx) =
            (0..levels.len())
                .filter(|i| !is_done(*i, &cursors))
                .min_by(|i1, i2| {
                    TotalOrd::tot_cmp(next_value(*i1, &cursors), next_value(*i2, &cursors))
                })
        {
            let level = levels[level_idx];
            let item = items[level.offset + cursors[level_idx]].clone();
            let weight = 2usize.pow(level_idx as u32);
            let cum_weight = *cum_weights.last().unwrap_or(&0) + weight;
            finalized_items.push(item);
            cum_weights.push(cum_weight);
            cursors[level_idx] += 1;
        }

        debug_assert_eq!(finalized_items.len(), items.len());
        debug_assert_eq!(cum_weights.len(), items.len());
        debug_assert_eq!(cum_weights.last().unwrap_or(&0), &self.consumed_items);

        FinalizedState {
            items: finalized_items.into_boxed_slice(),
            cum_weight: Some(cum_weights.into_boxed_slice()),
        }
    }
}

impl<T: fmt::Debug + Clone + TotalOrd> FinalizedState<T> {
    pub(super) fn new(items: Box<[T]>, cum_weight: Option<Box<[usize]>>) -> Self {
        Self { items, cum_weight }
    }

    pub(super) fn num_items(&self) -> usize {
        match &self.cum_weight {
            Some(cum_weight) => cum_weight.last().map(|x| *x).unwrap_or(0),
            None => self.items.len(),
        }
    }

    pub(super) fn estimate_rank(&self, value: &T) -> usize {
        todo!()
    }

    pub(super) fn estimate_quantile(&self, quantile: f64) -> &T {
        assert!(
            (0.0..=1.0).contains(&quantile),
            "quantile should be between 0.0 and 1.0"
        );
        let estimated_rank =
            (quantile * self.num_items().saturating_sub(1) as f64).round() as usize + 1;
        let idx = estimate_quantile_index(self.cum_weight.as_ref(), estimated_rank);
        &self.items[idx]
    }
}

#[inline(never)]
fn estimate_quantile_index(cum_weight: Option<&Box<[usize]>>, estimated_rank: usize) -> usize {
    match cum_weight {
        Some(cum_weight) => cum_weight.partition_point(|w| *w < estimated_rank),
        None => estimated_rank - 1,
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
