//! What a hash join knows about a build-side key, published to the scans
//! below its probe side once the build is done: the key's range, which scans
//! use to skip batches by their statistics, and optionally a bloom filter over
//! the keys, which scans probe per row.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use polars_arrow::bitmap::BitmapBuilder;
use polars_core::config;
use polars_core::prelude::*;
use polars_core::runtime::{ASYNC, RAYON};
use polars_expr::hash_keys::HashKeys;
use polars_io::predicates::{RuntimeRange, cast_bound};
use polars_plan::plans::options::{MAX_BUILD_PROBE_DISTINCT_RATIO, RuntimeFilter};
use polars_plan::plans::{PredicateExpr, TrivialPredicateExpr};
use polars_utils::bloom_filter::SplitBlockBloom;
use polars_utils::cardinality_sketch::CardinalitySketch;
use rayon::prelude::*;

use super::select_key_columns;
use crate::expression::StreamExpr;
use crate::morsel::Morsel;
use crate::nodes::ExecutionState;

/// Bits per key a bloom filter is sized for.
const BLOOM_BITS_PER_KEY: usize = 8;
/// A bloom filter with fewer bits per distinct key than this is not published.
const BLOOM_MIN_BITS_PER_KEY: usize = 4;
/// Smallest bloom filter that is built.
const BLOOM_MIN_BYTES: usize = 64 << 10;
/// Largest bloom filter that is published.
const BLOOM_MAX_BYTES: usize = 32 << 20;
/// Build rows whose key hashes are kept, over all builders of one filter, so
/// the bloom filter can be sized from the keys actually seen. Past this, the
/// bloom filter gets the size the plan estimated.
const BUFFERED_ROWS_BUDGET: usize = BLOOM_MAX_BYTES / size_of::<u64>();

/// The runtime filters of a join and how each is built. Published once, from
/// the side the plan named as build side. Builders come from `new_builders`
/// and hold one entry per filter, in the same order.
pub(super) struct RuntimeFilters {
    filters: Vec<(RuntimeFilter, KeyFilterSpec)>,
}

impl RuntimeFilters {
    /// `key_schema` holds the join keys by position; both sides have the same
    /// dtypes.
    pub(super) fn new(filters: Vec<RuntimeFilter>, key_schema: &Schema) -> Self {
        let filters = filters
            .into_iter()
            .map(|filter| {
                let spec = KeyFilterSpec::new(&filter, key_schema);
                (filter, spec)
            })
            .collect();
        Self { filters }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Whether the filters were published already.
    pub(super) fn is_set(&self) -> bool {
        self.filters.first().is_some_and(|(f, _)| f.pred.is_set())
    }

    pub(super) fn new_builders(&self) -> Vec<KeyFilterBuilder> {
        self.filters
            .iter()
            .map(|(_, spec)| KeyFilterBuilder::new(spec))
            .collect()
    }

    /// Add the key column of every filter to its builder.
    pub(super) fn extend(
        &self,
        keys: &DataFrame,
        builders: &mut [KeyFilterBuilder],
    ) -> PolarsResult<()> {
        assert_eq!(builders.len(), self.filters.len());
        for ((filter, _), builder) in self.filters.iter().zip(builders) {
            builder.extend(&keys.columns()[filter.key_idx])?;
        }
        Ok(())
    }

    /// Set every filter to what its builder collected.
    pub(super) fn publish(&self, builders: Vec<KeyFilterBuilder>) {
        assert_eq!(builders.len(), self.filters.len());
        for ((filter, _), builder) in self.filters.iter().zip(builders) {
            let key_filter = builder.finish();
            if config::verbose() {
                eprintln!(
                    "publishing runtime filter for key {}: {key_filter:?}",
                    filter.key_idx
                );
            }
            filter.pred.set(Arc::new(key_filter));
        }
    }

    /// Set every filter to the keys of a completely sampled side. The filter
    /// holds whichever side is built later, as no key of the other side outside
    /// it can match.
    pub(super) fn publish_from_sample(
        &self,
        morsels: &[Morsel],
        key_selectors: &[StreamExpr],
        state: &ExecutionState,
    ) -> PolarsResult<()> {
        let new_builders = || self.new_builders();
        let builders = RAYON.install(|| {
            morsels
                .par_iter()
                .try_fold(new_builders, |mut builders, morsel| {
                    let df = morsel.df_blocking();
                    let keys = ASYNC.block_on(select_key_columns(&df, key_selectors, state))?;
                    self.extend(&keys, &mut builders)?;
                    PolarsResult::Ok(builders)
                })
                .try_reduce(new_builders, |mut a, b| {
                    for (a, b) in a.iter_mut().zip(b) {
                        a.merge(b);
                    }
                    Ok(a)
                })
        })?;
        self.publish(builders);
        Ok(())
    }

    /// Set every filter to what the builders of every local build collected.
    pub(super) fn publish_merged(&self, locals: impl IntoIterator<Item = Vec<KeyFilterBuilder>>) {
        let mut builders = self.new_builders();
        for local in locals {
            assert_eq!(local.len(), builders.len());
            for (builder, seen) in builders.iter_mut().zip(local) {
                builder.merge(seen);
            }
        }
        self.publish(builders);
    }

    /// Set every filter not published yet to one that skips nothing.
    pub(super) fn publish_nothing(&self) {
        for (filter, _) in &self.filters {
            if !filter.pred.is_set() {
                filter.pred.set(Arc::new(TrivialPredicateExpr));
            }
        }
    }
}

/// How a join builds the filter of one key.
#[derive(Clone, Debug)]
struct KeyFilterSpec {
    dtype: DataType,
    /// Distinct keys the bloom filter is sized for; `None` gives the range only.
    bloom_keys: Option<usize>,
    /// The probe's distinct keys, when the plan could estimate them.
    probe_distinct: Option<usize>,
    /// Shared by the build and probe sides.
    random_state: PlRandomState,
    /// Build rows seen by all builders of this filter. It only grows.
    rows_seen: Arc<AtomicUsize>,
    /// `BUFFERED_ROWS_BUDGET`, lowered in tests.
    rows_budget: usize,
}

impl KeyFilterSpec {
    fn new(filter: &RuntimeFilter, key_schema: &Schema) -> Self {
        Self {
            dtype: key_schema.get_at_index(filter.key_idx).unwrap().1.clone(),
            bloom_keys: filter.bloom_keys,
            probe_distinct: filter.probe_distinct,
            random_state: PlRandomState::default(),
            rows_seen: Arc::default(),
            rows_budget: BUFFERED_ROWS_BUDGET,
        }
    }

    /// An empty bloom filter for `keys` distinct keys, or `None` when it would
    /// be larger than is published.
    fn bloom_for(keys: usize) -> Option<SplitBlockBloom> {
        let keys = keys.max(BLOOM_MIN_BYTES * 8 / BLOOM_BITS_PER_KEY);
        let bytes = SplitBlockBloom::size_for(keys, BLOOM_BITS_PER_KEY);
        (bytes <= BLOOM_MAX_BYTES).then(|| SplitBlockBloom::with_capacity(keys, BLOOM_BITS_PER_KEY))
    }

    /// The bloom filter of the size the plan estimated. Every call gives the
    /// same size, so any two can be merged.
    fn planned_bloom(&self) -> Option<SplitBlockBloom> {
        Self::bloom_for(self.bloom_keys?)
    }

    fn hash_keys(&self, column: &Column) -> HashKeys {
        HashKeys::from_df(
            &column.clone().into_frame(),
            self.random_state.clone(),
            false,
            false,
        )
    }
}

/// Collects the keys one builder sees for a `KeyFilter`.
pub(super) struct KeyFilterBuilder {
    range: KeyRange,
    bloom: Option<BloomBuilder>,
}

struct BloomBuilder {
    spec: KeyFilterSpec,
    keys: BloomKeys,
    sketch: CardinalitySketch,
}

enum BloomKeys {
    /// The key hashes, while the filter's build rows fit in
    /// `BUFFERED_ROWS_BUDGET`. The bloom filter is sized when publishing.
    Buffered(Vec<u64>),
    /// A bloom filter of the planned size, or `None` when that is too large.
    Planned(Option<SplitBlockBloom>),
}

impl BloomKeys {
    /// Move the buffered hashes into a bloom filter of the planned size.
    fn switch_to_planned(&mut self, spec: &KeyFilterSpec) {
        if let BloomKeys::Buffered(hashes) = self {
            if config::verbose() {
                eprintln!(
                    "runtime filter over {} build rows, using the planned bloom filter size",
                    spec.rows_budget
                );
            }
            let mut bloom = spec.planned_bloom();
            if let Some(bloom) = &mut bloom {
                for &hash in hashes.iter() {
                    bloom.insert(hash);
                }
            }
            *self = BloomKeys::Planned(bloom);
        }
    }
}

impl KeyFilterBuilder {
    fn new(spec: &KeyFilterSpec) -> Self {
        Self {
            range: KeyRange::default(),
            bloom: spec.bloom_keys.map(|_| BloomBuilder {
                spec: spec.clone(),
                keys: BloomKeys::Buffered(Vec::new()),
                sketch: CardinalitySketch::new(),
            }),
        }
    }

    /// Add the non-null values of `column`.
    fn extend(&mut self, column: &Column) -> PolarsResult<()> {
        self.range.extend(column)?;
        if let Some(b) = &mut self.bloom {
            let rows_seen = b.spec.rows_seen.fetch_add(column.len(), Ordering::Relaxed);
            if rows_seen + column.len() > b.spec.rows_budget {
                b.keys.switch_to_planned(&b.spec);
            }
            if matches!(b.keys, BloomKeys::Planned(None)) {
                return Ok(());
            }
            let hash_keys = b.spec.hash_keys(column);
            match &mut b.keys {
                BloomKeys::Buffered(hashes) => hash_keys.for_each_hash(|_, hash| {
                    if let Some(hash) = hash {
                        hashes.push(hash);
                        b.sketch.insert(hash);
                    }
                }),
                BloomKeys::Planned(Some(bloom)) => hash_keys.for_each_hash(|_, hash| {
                    if let Some(hash) = hash {
                        bloom.insert(hash);
                        b.sketch.insert(hash);
                    }
                }),
                BloomKeys::Planned(None) => unreachable!(),
            }
        }
        Ok(())
    }

    /// Add everything `other` collected.
    pub(super) fn merge(&mut self, other: Self) {
        self.range.merge(other.range);
        let (Some(a), Some(mut b)) = (&mut self.bloom, other.bloom) else {
            return;
        };
        a.sketch.combine(&b.sketch);
        if matches!(b.keys, BloomKeys::Planned(_)) {
            a.keys.switch_to_planned(&a.spec);
        }
        match (&mut a.keys, &mut b.keys) {
            (BloomKeys::Buffered(a), BloomKeys::Buffered(b)) => a.append(b),
            (BloomKeys::Planned(a), BloomKeys::Buffered(b)) => {
                if let Some(a) = a {
                    for &hash in b.iter() {
                        a.insert(hash);
                    }
                }
            },
            (BloomKeys::Planned(a), BloomKeys::Planned(b)) => {
                if let (Some(a), Some(b)) = (a, b) {
                    a.union_with(b);
                }
            },
            (BloomKeys::Buffered(_), BloomKeys::Planned(_)) => unreachable!(),
        }
    }

    /// The filter to publish. The bloom filter is left out when it holds too
    /// many distinct keys for its size, or too large a share of the probe's
    /// distinct keys to be worth probing.
    fn finish(self) -> KeyFilter {
        let bloom = self.bloom.and_then(|b| {
            let distinct = b.sketch.estimate();
            let weak = b
                .spec
                .probe_distinct
                .is_some_and(|probe| distinct as f64 > probe as f64 * MAX_BUILD_PROBE_DISTINCT_RATIO);
            let bloom = match b.keys {
                _ if weak => None,
                BloomKeys::Buffered(hashes) => {
                    KeyFilterSpec::bloom_for(distinct).map(|mut bloom| {
                        for hash in hashes {
                            bloom.insert(hash);
                        }
                        bloom
                    })
                },
                BloomKeys::Planned(bloom) => bloom.filter(|bloom| {
                    distinct.saturating_mul(BLOOM_MIN_BITS_PER_KEY) <= bloom.num_bits()
                }),
            };
            if bloom.is_none() && config::verbose() {
                eprintln!(
                    "dropping bloom filter: {distinct} distinct build keys, {:?} distinct probe keys",
                    b.spec.probe_distinct
                );
            }
            Some(KeyBloom {
                spec: b.spec,
                bloom: bloom?,
            })
        });
        KeyFilter {
            range: self.range,
            bloom,
        }
    }
}

/// The range of one build key column and, when it pays, a bloom filter over
/// its values.
#[derive(Clone, Debug)]
struct KeyFilter {
    range: KeyRange,
    bloom: Option<KeyBloom>,
}

#[derive(Clone)]
struct KeyBloom {
    spec: KeyFilterSpec,
    bloom: SplitBlockBloom,
}

impl std::fmt::Debug for KeyBloom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bloom of {} bytes", self.bloom.size_bytes())
    }
}

impl PredicateExpr for KeyFilter {
    /// Whether each value may be a build key. Nulls never are. A column of
    /// another dtype than the key hashes differently and is left alone.
    fn evaluate(&self, columns: &[Column]) -> PolarsResult<Option<Column>> {
        let Some(KeyBloom { spec, bloom }) = &self.bloom else {
            return Ok(None);
        };
        let column = &columns[0];
        if column.dtype() != &spec.dtype {
            return Ok(None);
        }
        let mut mask = BitmapBuilder::with_capacity(column.len());
        spec.hash_keys(column).for_each_hash(|_, hash| {
            mask.push(hash.is_some_and(|hash| bloom.contains(hash)));
        });
        Ok(Some(
            BooleanChunked::from_bitmap(column.name().clone(), mask.freeze()).into_column(),
        ))
    }

    fn evaluate_stats(
        &self,
        min: &Column,
        max: &Column,
        null_count: &Column,
    ) -> PolarsResult<Option<Column>> {
        self.range.evaluate_stats(min, max, null_count)
    }

    fn runtime_range(&self) -> RuntimeRange {
        self.range.runtime_range()
    }

    fn filters_rows(&self) -> bool {
        self.bloom.is_some()
    }

    fn can_bypass(&self) -> bool {
        true
    }
}

/// Min and max of one build key column. Empty until a non-null key is seen; an
/// empty range published after the build means nothing can match.
#[derive(Clone, Debug, Default)]
struct KeyRange {
    bounds: Option<(Scalar, Scalar)>,
}

impl KeyRange {
    /// Widen the range to cover the non-null values of `column`.
    fn extend(&mut self, column: &Column) -> PolarsResult<()> {
        let min = column.min_reduce()?;
        let max = column.max_reduce()?;
        if !min.is_null() && !max.is_null() {
            self.merge(Self {
                bounds: Some((min, max)),
            });
        }
        Ok(())
    }

    /// Widen the range to cover `other`.
    fn merge(&mut self, other: Self) {
        let Some((min, max)) = other.bounds else {
            return;
        };
        match &mut self.bounds {
            None => self.bounds = Some((min, max)),
            Some((lo, hi)) => {
                if min.value() < lo.value() {
                    *lo = min;
                }
                if max.value() > hi.value() {
                    *hi = max;
                }
            },
        }
    }
}

/// The bounds as length-one series of `dtype`, or `None` when a bound does not
/// survive the cast, in which case the range says nothing.
fn bounds_for(bounds: &(Scalar, Scalar), dtype: &DataType) -> Option<(Series, Series)> {
    let lo = cast_bound(&bounds.0, dtype)?;
    let hi = cast_bound(&bounds.1, dtype)?;
    Some((
        lo.into_series(PlSmallStr::EMPTY),
        hi.into_series(PlSmallStr::EMPTY),
    ))
}

impl PredicateExpr for KeyRange {
    fn evaluate_stats(
        &self,
        min: &Column,
        max: &Column,
        _null_count: &Column,
    ) -> PolarsResult<Option<Column>> {
        let Some(bounds) = &self.bounds else {
            return Ok(Some(Column::new_scalar(
                min.name().clone(),
                Scalar::from(true),
                min.len(),
            )));
        };
        let min = min.as_materialized_series();
        let max = max.as_materialized_series();
        let Some((lo, hi)) = bounds_for(bounds, min.dtype()) else {
            return Ok(None);
        };
        // A batch is skipped when it lies entirely below or above the range. An
        // unknown statistic is null and settles nothing.
        let skip = max.lt(&lo)? | min.gt(&hi)?;
        Ok(Some(skip.fill_null_with_values(false)?.into_column()))
    }

    fn runtime_range(&self) -> RuntimeRange {
        match &self.bounds {
            None => RuntimeRange::Empty,
            Some((lo, hi)) => RuntimeRange::Range {
                lo: lo.clone(),
                hi: hi.clone(),
            },
        }
    }

    fn filters_rows(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(bloom_keys: usize, rows_budget: usize) -> KeyFilterSpec {
        spec_with_hashes(bloom_keys, rows_budget, PlRandomState::default())
    }

    /// A spec with its own row count, hashing like every other spec given
    /// `random_state`, so their filters can be compared bit for bit.
    fn spec_with_hashes(
        bloom_keys: usize,
        rows_budget: usize,
        random_state: PlRandomState,
    ) -> KeyFilterSpec {
        KeyFilterSpec {
            dtype: DataType::Int64,
            bloom_keys: Some(bloom_keys),
            probe_distinct: None,
            random_state,
            rows_seen: Arc::default(),
            rows_budget,
        }
    }

    fn keys(range: std::ops::Range<i64>) -> Column {
        Column::new("k".into(), range.collect::<Vec<_>>())
    }

    fn is_buffered(builder: &KeyFilterBuilder) -> bool {
        matches!(builder.bloom.as_ref().unwrap().keys, BloomKeys::Buffered(_))
    }

    /// The bloom filter's size and which of `probe` it may contain.
    fn published(builder: KeyFilterBuilder, probe: &Column) -> Option<(usize, Vec<bool>)> {
        let filter = builder.finish();
        let size = filter.bloom.as_ref()?.bloom.size_bytes();
        let mask = filter
            .evaluate(std::slice::from_ref(probe))
            .unwrap()
            .unwrap();
        let mask = mask.bool().unwrap().iter().map(|m| m.unwrap()).collect();
        Some((size, mask))
    }

    #[test]
    fn buffered_bloom_is_sized_from_the_keys_seen() {
        // Planned for 1000 keys, but 1M arrive.
        let spec = spec(1000, usize::MAX);
        let mut builder = KeyFilterBuilder::new(&spec);
        builder.extend(&keys(0..1_000_000)).unwrap();
        assert!(is_buffered(&builder));

        let (size, mask) = published(builder, &keys(0..1_000_000)).unwrap();
        assert!(size * 8 >= 1_000_000 * BLOOM_MIN_BITS_PER_KEY);
        assert!(mask.iter().all(|m| *m));
    }

    #[test]
    fn planned_bloom_past_the_budget() {
        let filled = spec(1000, 10_000);
        let mut builder = KeyFilterBuilder::new(&filled);
        builder.extend(&keys(0..5_000)).unwrap();
        assert!(is_buffered(&builder));
        builder.extend(&keys(5_000..10_001)).unwrap();
        assert!(!is_buffered(&builder));

        // The planned size holds these keys, so it is published.
        let (size, mask) = published(builder, &keys(0..10_001)).unwrap();
        assert_eq!(size, BLOOM_MIN_BYTES);
        assert!(mask.iter().all(|m| *m));

        // Too many keys for the planned size, so it is dropped.
        let overloaded = spec(1000, 10_000);
        let mut builder = KeyFilterBuilder::new(&overloaded);
        builder.extend(&keys(0..1_000_000)).unwrap();
        assert!(published(builder, &keys(0..10)).is_none());
    }

    #[test]
    fn budget_is_shared_by_all_builders() {
        let spec = spec(1000, 1000);
        let mut a = KeyFilterBuilder::new(&spec);
        let mut b = KeyFilterBuilder::new(&spec);
        a.extend(&keys(0..600)).unwrap();
        b.extend(&keys(600..1200)).unwrap();
        assert!(is_buffered(&a));
        assert!(!is_buffered(&b));
        // `a` goes over on its next batch, however small.
        a.extend(&keys(1200..1201)).unwrap();
        assert!(!is_buffered(&a));
    }

    #[test]
    fn skew_under_the_budget_stays_buffered() {
        let probe = keys(0..20_000);
        let random_state = PlRandomState::default();
        let build = |splits: &[std::ops::Range<i64>]| {
            let spec = spec_with_hashes(10, 100_000, random_state.clone());
            let mut merged = KeyFilterBuilder::new(&spec);
            for range in splits {
                let mut local = KeyFilterBuilder::new(&spec);
                local.extend(&keys(range.clone())).unwrap();
                assert!(is_buffered(&local));
                merged.merge(local);
            }
            assert!(is_buffered(&merged));
            published(merged, &probe).unwrap()
        };
        let skewed = build(&[0..90_000, 90_000..95_000, 95_000..100_000]);
        let even = build(&[0..33_000, 33_000..66_000, 66_000..100_000]);
        assert_eq!(skewed, even);
    }

    #[test]
    fn merge_order_does_not_matter() {
        let probe = keys(0..40_000);
        // Two builders of one filter, one over the budget.
        let random_state = PlRandomState::default();
        let pair = || {
            let spec = spec_with_hashes(1000, 5_000, random_state.clone());
            let mut buffered = KeyFilterBuilder::new(&spec);
            buffered.extend(&keys(0..4_000)).unwrap();
            let mut planned = KeyFilterBuilder::new(&spec);
            planned.extend(&keys(4_000..8_000)).unwrap();
            assert!(is_buffered(&buffered));
            assert!(!is_buffered(&planned));
            (spec, buffered, planned)
        };

        let (spec, buffered, planned) = pair();
        let mut ab = KeyFilterBuilder::new(&spec);
        ab.merge(buffered);
        ab.merge(planned);

        let (spec, buffered, planned) = pair();
        let mut ba = KeyFilterBuilder::new(&spec);
        ba.merge(planned);
        ba.merge(buffered);

        let (_, buffered, mut planned) = pair();
        planned.merge(buffered);

        let ab = published(ab, &probe).unwrap();
        assert!(ab.1[..8_000].iter().all(|m| *m));
        assert_eq!(ab, published(ba, &probe).unwrap());
        assert_eq!(ab, published(planned, &probe).unwrap());
    }
}
