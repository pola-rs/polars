use std::{fmt, mem};

use polars_utils::total_ord::TotalOrd;
use rand::RngExt;
use rand::rngs::SmallRng;

// TODO: [amber] Reseed on clone()

/// KLL calls this `δ`. Equivalent to a 99.9999% success rate per queried value.
// const FAILURE_PROBABILITY: f64 = 1e-6;
const FAILURE_PROBABILITY: f64 = 0.5;

fn compute_k(error: f64, failure_prob: f64, n: usize) -> usize {
    assert!(error > 0.0 && error < 1.0, "invalid error: {error}");
    assert!(
        failure_prob > 0.0 && failure_prob <= 0.5,
        "invalid failure probability: {failure_prob}"
    );

    // Eq. 6
    2 * f64::ceil((4.0 / error) * f64::sqrt((-f64::ln(failure_prob)) / f64::log2(error * n as f64)))
        as usize
}

#[derive(Debug, Clone, Copy)]
struct Level {
    compaction_schedule: u64,
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
    levels: Vec<Level>,
    n: usize,
    k: usize,
    b: usize,
    /// List of relative compactors.
    rel_compactors: Vec<Vec<T>>,
    consumed_items: usize,
    rng: SmallRng,
}

#[derive(Debug, Clone, Default)]
struct FinalizedState<T: fmt::Debug + Clone + TotalOrd> {
    items: Box<[T]>,
    cum_weight: Option<Box<[usize]>>,
}

#[derive(Debug, Clone)]
enum State<T: fmt::Debug + Clone + TotalOrd> {
    Ingesting(IngestingState<T>),
    Finalized(FinalizedState<T>),
}

#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct ReqSketchBounded<T: fmt::Debug + Clone + TotalOrd>(State<T>);

impl<T: fmt::Debug + Clone + TotalOrd> ReqSketchBounded<T> {
    pub fn new(error: f64, n: usize) -> Self {
        let k = compute_k(error, FAILURE_PROBABILITY, n);
        assert!(n > k, "n must be greater than k");
        let state = IngestingState {
            levels: vec![Level {
                compaction_schedule: 0,
            }],
            n,
            k,
            b: dbg!(compactor_threshold_b(k, n)),
            rel_compactors: vec![Vec::new()],
            consumed_items: 0,
            rng: rand::make_rng(),
        };
        ReqSketchBounded(State::Ingesting(state))
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
        for item in array {
            assert!(self.space_left() > 0);
            self.compact_if_needed(0);
            self.rel_compactors[0].push(item.clone());
            self.consumed_items += 1;
        }
    }

    fn space_left(&self) -> usize {
        self.n.saturating_sub(self.consumed_items)
    }

    fn is_compactor_full(&self, level: usize) -> bool {
        self.rel_compactors[level].len() >= self.b
    }

    /// Compact all of the compactors from base to top.
    fn compact_if_needed(&mut self, level: usize) {
        while self.is_compactor_full(level) {
            let old_size = self.rel_compactors[level].len();
            self.compact_level_once(level);
            debug_assert!(self.rel_compactors[level].len() < old_size);
        }
        debug_assert!(self.rel_compactors[level].len() < self.b);
    }

    fn add_new_compactor(&mut self) {
        self.levels.push(Level {
            compaction_schedule: 0,
        });
        self.rel_compactors.push(Vec::new());
    }

    fn compact_level_once(&mut self, level: usize) {
        if level == self.levels.len() - 1 {
            self.add_new_compactor();
        }
        let [cur_level, next_level] = self
            .rel_compactors
            .get_disjoint_mut([level, level + 1])
            .unwrap();
        debug_assert!(cur_level.len() >= self.b, "compactor is not full");

        // Compute L_C
        let z_c = self.levels[level].compaction_schedule.trailing_ones();
        let l_c = (z_c as usize + 1) * self.k;
        let s_c = self.b - l_c;
        self.levels[level].compaction_schedule += 1;

        let rand: u8 = self.rng.random();
        let coin = rand & 0x1 != 0;

        // select_nth_unstable etc.
        cur_level[..self.b].select_nth_unstable_by(s_c, TotalOrd::tot_cmp);
        cur_level[s_c..self.b].sort_unstable_by(TotalOrd::tot_cmp);
        let mut drain = cur_level.drain(s_c..self.b);
        debug_assert!(drain.len() <= self.b / 2);
        debug_assert!(drain.len() % 2 == 0);

        // Throw away half of the values during the compaction
        if coin {
            drain.next();
        }
        let iter = drain.step_by(2);

        // Merge the items into the next compactor
        next_level.extend(iter);
        self.compact_if_needed(level + 1);
    }

    fn finalize(self) -> FinalizedState<T> {
        let IngestingState {
            mut rel_compactors,
            consumed_items,
            ..
        } = self;

        let pool_size: usize = rel_compactors.iter().map(|c| c.len()).sum();
        dbg!(&pool_size);

        // Compaction only partially orders a compactor, so sort them all.
        for compactor in rel_compactors.iter_mut() {
            compactor.sort_unstable_by(TotalOrd::tot_cmp);
        }

        // With a single compactor every item has weight 1.
        if rel_compactors.len() <= 1 {
            let items = rel_compactors.pop().unwrap_or_default();
            return FinalizedState {
                items: items.into_boxed_slice(),
                cum_weight: None,
            };
        }

        // Merge all sorted levels
        let num_items: usize = rel_compactors.iter().map(Vec::len).sum();
        let mut finalized_items = Vec::with_capacity(num_items);
        let mut cum_weights = Vec::with_capacity(num_items);
        let mut cursors: Vec<usize> = vec![0; rel_compactors.len()];

        // Are we done draining this level?
        let is_done =
            |level: usize, cursors: &[usize]| cursors[level] >= rel_compactors[level].len();
        // Get the next value corresponding to level `level`.
        let next_value = |level: usize, cursors: &[usize]| &rel_compactors[level][cursors[level]];

        // H-way merge-sort
        while let Some(level_idx) = (0..rel_compactors.len())
            .filter(|i| !is_done(*i, &cursors))
            .min_by(|i1, i2| {
                TotalOrd::tot_cmp(next_value(*i1, &cursors), next_value(*i2, &cursors))
            })
        {
            let item = rel_compactors[level_idx][cursors[level_idx]].clone();
            let weight = 1usize << level_idx;
            let cum_weight = cum_weights.last().unwrap_or(&0) + weight;
            finalized_items.push(item);
            cum_weights.push(cum_weight);
            cursors[level_idx] += 1;
        }

        debug_assert_eq!(finalized_items.len(), num_items);
        debug_assert_eq!(cum_weights.len(), num_items);
        debug_assert_eq!(cum_weights.last().unwrap_or(&0), &consumed_items);

        FinalizedState {
            items: finalized_items.into_boxed_slice(),
            cum_weight: Some(cum_weights.into_boxed_slice()),
        }
    }
}

impl<T: fmt::Debug + Clone + TotalOrd> FinalizedState<T> {
    fn num_items(&self) -> usize {
        match &self.cum_weight {
            Some(cum_weight) => cum_weight.last().map(|x| *x).unwrap_or(0),
            None => self.items.len(),
        }
    }

    fn estimate_rank(&self, value: &T) -> usize {
        todo!()
    }

    fn estimate_quantile(&self, quantile: f64) -> &T {
        assert!(
            (0.0..=1.0).contains(&quantile),
            "invalid quantile: {quantile}"
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

fn compactor_threshold_b(k: usize, n: usize) -> usize {
    // Sec 2.1: k is an *even* integer parameter.
    assert!(k > 0 && k % 2 == 0, "k must be a positive even integer");
    // dbg!(&k);
    // dbg!(&n);
    // dbg!(2 * k * (usize::div_ceil(n, k) * 2 - 1).ilog2() as usize);
    2 * k * (usize::div_ceil(n, k) * 2 - 1).ilog2() as usize
}
