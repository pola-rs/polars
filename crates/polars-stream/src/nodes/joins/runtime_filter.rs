//! What a hash join knows about a build-side key, published to the scans
//! below its probe side once the build is done: the key's range, which scans
//! use to skip batches by their statistics and to drop rows, and optionally a
//! bloom filter over the keys, which scans probe per row instead. Integer keys
//! in a small range get an exact bitset instead of a bloom filter.

use std::sync::Arc;

use num_traits::ToPrimitive;
use parking_lot::Mutex;
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_core::prelude::*;
use polars_core::runtime::{ASYNC, RAYON};
use polars_core::{config, with_match_physical_integer_polars_type};
use polars_expr::hash_keys::HashKeys;
use polars_io::predicates::{RuntimeRange, cast_bound};
use polars_ooc::memory_manager;
use polars_plan::plans::options::{MAX_BUILD_PROBE_DISTINCT_RATIO, RuntimeFilter};
use polars_plan::plans::{PredicateExpr, TrivialPredicateExpr};
use polars_utils::bloom_filter::{AtomicSplitBlockBloom, SplitBlockBloom};
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::relaxed_cell::RelaxedCell;
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
/// Largest key range a bitset is built for.
const BITSET_MAX_BITS: u64 = 64 << 20;
/// A bitset up to this size is taken even when a bloom filter would be smaller,
/// as it needs no hashing.
const BITSET_MIN_BYTES: usize = 1 << 20;
/// Build rows whose key hashes are kept, over all builders of one filter, so
/// the bloom filter can be sized from the keys seen. Past this, the builders
/// share one bloom filter of the size the plan estimated.
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

    /// Set every filter to the keys of a completely sampled side. The filter
    /// holds whichever side is built later, as no key of the other side outside
    /// it can match.
    pub(super) fn publish_from_sample(
        &self,
        morsels: &[Morsel],
        key_selectors: &[StreamExpr],
        state: &ExecutionState,
    ) -> PolarsResult<()> {
        // One set of builders per thread.
        let chunk_len = morsels.len().div_ceil(RAYON.current_num_threads()).max(1);
        let locals = RAYON.install(|| {
            morsels
                .par_chunks(chunk_len)
                .map(|chunk| {
                    let mut builders = self.new_builders();
                    for morsel in chunk {
                        {
                            let df = morsel.df_blocking();
                            let keys =
                                ASYNC.block_on(select_key_columns(&df, key_selectors, state))?;
                            self.extend(&keys, &mut builders)?;
                        }
                        memory_manager().spill_blocking();
                    }
                    PolarsResult::Ok(builders)
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;
        self.publish_merged(locals);
        Ok(())
    }

    /// Set every filter to what the builders of every local build collected.
    pub(super) fn publish_merged(&self, locals: impl IntoIterator<Item = Vec<KeyFilterBuilder>>) {
        let mut per_filter: Vec<Vec<KeyFilterBuilder>> =
            self.filters.iter().map(|_| Vec::new()).collect();
        for local in locals {
            assert_eq!(local.len(), per_filter.len());
            for (builders, builder) in per_filter.iter_mut().zip(local) {
                builders.push(builder);
            }
        }
        for ((filter, spec), builders) in self.filters.iter().zip(per_filter) {
            let key_filter = KeyFilter::from_builders(spec, builders);
            if config::verbose() {
                eprintln!(
                    "publishing runtime filter for key {}: {key_filter:?}",
                    filter.key_idx
                );
            }
            filter.pred.set(Arc::new(key_filter));
        }
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
#[derive(Clone)]
struct KeyFilterSpec {
    dtype: DataType,
    /// Distinct keys the bloom filter is sized for; `None` gives the range only.
    bloom_keys: Option<usize>,
    /// The probe's distinct keys, when the plan could estimate them.
    probe_distinct: Option<usize>,
    /// Shared by the build and probe sides.
    random_state: PlRandomState,
    /// Integer keys, which can get a bitset.
    int_keys: bool,
    shared: Arc<SharedBloom>,
}

/// What all builders of one filter share.
#[derive(Default)]
struct SharedBloom {
    /// Build rows seen by all builders. It only grows.
    rows_seen: RelaxedCell<usize>,
    /// The bloom filter of the planned size, made by the first builder past
    /// `BUFFERED_ROWS_BUDGET` and filled by all of them. Taken when publishing.
    planned: Mutex<Option<Arc<AtomicSplitBlockBloom>>>,
}

impl KeyFilterSpec {
    fn new(filter: &RuntimeFilter, key_schema: &Schema) -> Self {
        let dtype = key_schema.get_at_index(filter.key_idx).unwrap().1.clone();
        Self {
            int_keys: dtype.is_integer(),
            dtype,
            bloom_keys: filter.bloom_keys,
            probe_distinct: filter.probe_distinct,
            random_state: PlRandomState::default(),
            shared: Arc::default(),
        }
    }

    /// The keys a bloom filter for `keys` distinct keys is sized for.
    fn bloom_capacity(keys: usize) -> usize {
        keys.max(BLOOM_MIN_BYTES * 8 / BLOOM_BITS_PER_KEY)
    }

    /// An empty bloom filter for `keys` distinct keys, or `None` when it would
    /// be larger than is published.
    fn bloom_for(keys: usize) -> Option<SplitBlockBloom> {
        let keys = Self::bloom_capacity(keys);
        (SplitBlockBloom::size_for(keys, BLOOM_BITS_PER_KEY) <= BLOOM_MAX_BYTES)
            .then(|| SplitBlockBloom::with_capacity(keys, BLOOM_BITS_PER_KEY))
    }

    /// The bloom filter of the size the plan estimated, which all builders
    /// share, or `None` when it would be larger than is published.
    fn planned_bloom(&self) -> Option<Arc<AtomicSplitBlockBloom>> {
        let mut planned = self.shared.planned.lock();
        if planned.is_none() {
            *planned = Self::bloom_for(self.bloom_keys?).map(|bloom| Arc::new(bloom.into()));
            if planned.is_some() && config::verbose() {
                eprintln!(
                    "runtime filter over {BUFFERED_ROWS_BUDGET} build rows, using the planned bloom filter size"
                );
            }
        }
        planned.clone()
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
    /// Every key hash seen while buffered, else those of the last batch.
    hashes: Vec<u64>,
    /// Every integer key column seen while buffered.
    columns: Vec<Column>,
    sketch: CardinalitySketch,
}

enum BloomKeys {
    /// The key hashes are kept while the filter's build rows fit in
    /// `BUFFERED_ROWS_BUDGET`. The bloom filter is sized when publishing.
    Buffered,
    /// The shared bloom filter of the planned size, or `None` when that is too
    /// large.
    Planned(Option<Arc<AtomicSplitBlockBloom>>),
}

impl BloomBuilder {
    /// Move the buffered hashes into the shared bloom filter of the planned
    /// size.
    fn switch_to_planned(&mut self) {
        if matches!(self.keys, BloomKeys::Buffered) {
            let bloom = self.spec.planned_bloom();
            if let Some(bloom) = &bloom {
                bloom.insert_many(&self.hashes);
            }
            self.hashes = Vec::new();
            self.columns = Vec::new();
            self.keys = BloomKeys::Planned(bloom);
        }
    }
}

impl KeyFilterBuilder {
    fn new(spec: &KeyFilterSpec) -> Self {
        Self {
            range: KeyRange::default(),
            bloom: spec.bloom_keys.map(|_| BloomBuilder {
                spec: spec.clone(),
                keys: BloomKeys::Buffered,
                hashes: Vec::new(),
                columns: Vec::new(),
                sketch: CardinalitySketch::new(),
            }),
        }
    }

    /// Add the non-null values of `column`.
    fn extend(&mut self, column: &Column) -> PolarsResult<()> {
        self.range.extend(column)?;
        if let Some(b) = &mut self.bloom {
            let rows_seen = b.spec.shared.rows_seen.fetch_add(column.len()) + column.len();
            if rows_seen > BUFFERED_ROWS_BUDGET {
                b.switch_to_planned();
            }
            let bloom = match &b.keys {
                BloomKeys::Buffered => {
                    if b.spec.int_keys {
                        b.columns.push(column.clone());
                    }
                    None
                },
                BloomKeys::Planned(None) => return Ok(()),
                BloomKeys::Planned(Some(bloom)) => {
                    b.hashes.clear();
                    Some(bloom)
                },
            };
            b.hashes.reserve(column.len());
            b.spec.hash_keys(column).for_each_hash(|_, hash| {
                if let Some(hash) = hash {
                    b.hashes.push(hash);
                    b.sketch.insert(hash);
                }
            });
            if let Some(bloom) = bloom {
                bloom.insert_many(&b.hashes);
            }
        }
        Ok(())
    }
}

impl KeyFilter {
    /// The filter to publish from all builders of `spec`. The bloom filter is
    /// left out when it holds too many distinct keys for its size, or too large
    /// a share of the probe's distinct keys to be worth probing.
    fn from_builders(spec: &KeyFilterSpec, builders: Vec<KeyFilterBuilder>) -> Self {
        let mut range = KeyRange::default();
        let mut sketch = CardinalitySketch::new();
        let mut planned = false;
        let mut buffered = Vec::new();
        let mut columns = Vec::new();
        for builder in builders {
            range.merge(builder.range);
            if let Some(b) = builder.bloom {
                sketch.combine(&b.sketch);
                match b.keys {
                    BloomKeys::Buffered => {
                        buffered.push(b.hashes);
                        columns.extend(b.columns);
                    },
                    BloomKeys::Planned(_) => planned = true,
                }
            }
        }
        // Taken whether it is published or not, so it is freed.
        let shared = spec.shared.planned.lock().take();
        if spec.bloom_keys.is_none() {
            return Self::range_only(range);
        }

        let distinct = sketch.estimate();
        let weak = spec
            .probe_distinct
            .is_some_and(|probe| distinct as f64 > probe as f64 * MAX_BUILD_PROBE_DISTINCT_RATIO);
        if !weak && spec.int_keys && !planned {
            let max_bytes = SplitBlockBloom::size_for(
                KeyFilterSpec::bloom_capacity(distinct),
                BLOOM_BITS_PER_KEY,
            )
            .max(BITSET_MIN_BYTES);
            if let Some(bitset) = KeyBitset::new(&spec.dtype, &range, &columns, max_bytes) {
                // The range alone is exact when every value in it is a key.
                let bitset = (!bitset.is_full()).then_some(bitset);
                return Self {
                    range,
                    bloom: None,
                    bitset,
                };
            }
        }
        let bloom = if weak {
            None
        } else if planned {
            // Every builder holding it was dropped above, so no thread inserts
            // anymore.
            shared
                .and_then(|bloom| {
                    debug_assert_eq!(Arc::strong_count(&bloom), 1);
                    Arc::into_inner(bloom).map(AtomicSplitBlockBloom::into_inner)
                })
                .filter(|bloom| distinct.saturating_mul(BLOOM_MIN_BITS_PER_KEY) <= bloom.num_bits())
        } else {
            KeyFilterSpec::bloom_for(distinct)
        };
        let bloom = bloom.map(|mut bloom| {
            for hashes in &buffered {
                bloom.insert_many(hashes);
            }
            bloom
        });
        if bloom.is_none() && config::verbose() {
            eprintln!(
                "dropping bloom filter: {distinct} distinct build keys, {:?} distinct probe keys",
                spec.probe_distinct
            );
        }
        Self {
            range,
            bloom: bloom.map(|bloom| KeyBloom {
                spec: spec.clone(),
                bloom,
            }),
            bitset: None,
        }
    }

    fn range_only(range: KeyRange) -> Self {
        Self {
            range,
            bloom: None,
            bitset: None,
        }
    }
}

/// The range of one build key column and, when it pays, a bloom filter or a
/// bitset over its values.
#[derive(Clone, Debug)]
struct KeyFilter {
    range: KeyRange,
    bloom: Option<KeyBloom>,
    bitset: Option<KeyBitset>,
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
    /// Whether each value may be a build key. Nulls never are. Without a bloom
    /// filter only the range is checked. A column of another dtype than the key
    /// hashes differently and is left alone.
    fn evaluate(&self, columns: &[Column]) -> PolarsResult<Option<Column>> {
        if let Some(bitset) = &self.bitset {
            return Ok(bitset.evaluate(&columns[0]));
        }
        let Some(KeyBloom { spec, bloom }) = &self.bloom else {
            return self.range.evaluate(columns);
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
        self.bloom.is_some() || self.bitset.is_some() || self.range.filters_rows()
    }

    fn can_bypass(&self) -> bool {
        true
    }
}

/// The integer keys in `min..min + width`, one bit each.
#[derive(Clone)]
struct KeyBitset {
    dtype: DataType,
    min: i64,
    width: u64,
    words: Vec<u64>,
}

impl std::fmt::Debug for KeyBitset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bitset of {} bytes", self.words.len() * size_of::<u64>())
    }
}

impl KeyBitset {
    /// The bitset of the keys in `columns`, or `None` when their range does not
    /// fit in `max_bytes`.
    fn new(
        dtype: &DataType,
        range: &KeyRange,
        columns: &[Column],
        max_bytes: usize,
    ) -> Option<Self> {
        let (lo, hi) = range.bounds.as_ref()?;
        let min = lo.value().extract::<i64>()?;
        let max = hi.value().extract::<i64>()?;
        let width = (max as i128 - min as i128 + 1) as u64;
        if width > BITSET_MAX_BITS || width.div_ceil(8) > max_bytes as u64 {
            return None;
        }
        let mut bitset = Self {
            dtype: dtype.clone(),
            min,
            width,
            words: vec![0; width.div_ceil(64) as usize],
        };
        for column in columns {
            let s = column.as_materialized_series().to_physical_repr();
            with_match_physical_integer_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                for v in ca.iter().flatten() {
                    let offset = v.to_i64()?.wrapping_sub(min) as u64;
                    bitset.words[(offset / 64) as usize] |= 1 << (offset % 64);
                }
            });
        }
        Some(bitset)
    }

    fn is_full(&self) -> bool {
        let ones: u64 = self.words.iter().map(|w| w.count_ones() as u64).sum();
        ones == self.width
    }

    #[inline(always)]
    fn contains(&self, v: Option<i64>) -> bool {
        let Some(v) = v else {
            return false;
        };
        let offset = v.wrapping_sub(self.min) as u64;
        // SAFETY: an offset below `width` is within `words`.
        offset < self.width
            && unsafe { self.words.get_unchecked((offset / 64) as usize) } >> (offset % 64) & 1 != 0
    }

    /// Which of `values` are keys.
    fn mask<N: ToPrimitive>(&self, values: &[N]) -> Bitmap {
        let bytes = values
            .chunks(8)
            .map(|chunk| {
                chunk.iter().enumerate().fold(0u8, |byte, (i, v)| {
                    byte | ((self.contains(v.to_i64()) as u8) << i)
                })
            })
            .collect();
        Bitmap::try_new(bytes, values.len()).unwrap()
    }

    /// Whether each value is a key. Nulls never are. A column of another dtype
    /// than the keys is left alone.
    fn evaluate(&self, column: &Column) -> Option<Column> {
        if column.dtype() != &self.dtype {
            return None;
        }
        let s = column.as_materialized_series().to_physical_repr();
        let mut mask = BitmapBuilder::with_capacity(column.len());
        with_match_physical_integer_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            for arr in ca.downcast_iter() {
                let keys = self.mask(arr.values().as_slice());
                match arr.validity() {
                    Some(validity) => mask.extend_from_bitmap(&(&keys & validity)),
                    None => mask.extend_from_bitmap(&keys),
                }
            }
        });
        Some(BooleanChunked::from_bitmap(column.name().clone(), mask.freeze()).into_column())
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
    /// Whether each value lies in the range. Nulls never do.
    fn evaluate(&self, columns: &[Column]) -> PolarsResult<Option<Column>> {
        let Some(bounds) = &self.bounds else {
            return Ok(None);
        };
        let column = columns[0].as_materialized_series();
        let Some((lo, hi)) = bounds_for(bounds, column.dtype()) else {
            return Ok(None);
        };
        let mask = (column.gt_eq(&lo)? & column.lt_eq(&hi)?).fill_null_with_values(false)?;
        Ok(Some(mask.into_column()))
    }

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
                lo: Some(lo.clone()),
                hi: Some(hi.clone()),
                nulls_match: false,
            },
        }
    }

    fn filters_rows(&self) -> bool {
        self.bounds.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A spec whose builders keep hashes for `rows_left` more rows.
    fn spec(bloom_keys: usize, rows_left: usize) -> KeyFilterSpec {
        spec_with_hashes(bloom_keys, rows_left, PlRandomState::default())
    }

    /// A spec with its own row count, hashing like every other spec given
    /// `random_state`, so their filters can be compared bit for bit.
    fn spec_with_hashes(
        bloom_keys: usize,
        rows_left: usize,
        random_state: PlRandomState,
    ) -> KeyFilterSpec {
        KeyFilterSpec {
            dtype: DataType::Int64,
            bloom_keys: Some(bloom_keys),
            probe_distinct: None,
            random_state,
            int_keys: false,
            shared: Arc::new(SharedBloom {
                rows_seen: RelaxedCell::from(BUFFERED_ROWS_BUDGET - rows_left),
                planned: Mutex::default(),
            }),
        }
    }

    fn keys(range: std::ops::Range<i64>) -> Column {
        Column::new("k".into(), range.collect::<Vec<_>>())
    }

    /// A builder of `spec` that saw the keys of `range` in one batch.
    fn built(spec: &KeyFilterSpec, range: std::ops::Range<i64>) -> KeyFilterBuilder {
        let mut builder = KeyFilterBuilder::new(spec);
        builder.extend(&keys(range)).unwrap();
        builder
    }

    fn planned_bloom(builder: &KeyFilterBuilder) -> Option<&Arc<AtomicSplitBlockBloom>> {
        match &builder.bloom.as_ref().unwrap().keys {
            BloomKeys::Planned(bloom) => bloom.as_ref(),
            BloomKeys::Buffered => None,
        }
    }

    fn is_buffered(builder: &KeyFilterBuilder) -> bool {
        matches!(builder.bloom.as_ref().unwrap().keys, BloomKeys::Buffered)
    }

    /// The bloom filter's size and which of `probe` it may contain.
    fn published(
        spec: &KeyFilterSpec,
        builders: Vec<KeyFilterBuilder>,
        probe: &Column,
    ) -> Option<(usize, Vec<bool>)> {
        let filter = KeyFilter::from_builders(spec, builders);
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
        let spec = spec(1000, BUFFERED_ROWS_BUDGET);
        let builder = built(&spec, 0..1_000_000);
        assert!(is_buffered(&builder));

        let (size, mask) = published(&spec, vec![builder], &keys(0..1_000_000)).unwrap();
        assert!(size * 8 >= 1_000_000 * BLOOM_MIN_BITS_PER_KEY);
        assert!(mask.iter().all(|m| *m));
    }

    #[test]
    fn planned_bloom_past_the_budget() {
        let filled = spec(1000, 10_000);
        let mut builder = built(&filled, 0..5_000);
        assert!(is_buffered(&builder));
        builder.extend(&keys(5_000..10_001)).unwrap();
        assert!(!is_buffered(&builder));

        // The planned size holds these keys, so it is published.
        let (size, mask) = published(&filled, vec![builder], &keys(0..10_001)).unwrap();
        assert_eq!(size, BLOOM_MIN_BYTES);
        assert!(mask.iter().all(|m| *m));

        // Too many keys for the planned size, so it is dropped.
        let overloaded = spec(1000, 10_000);
        let builder = built(&overloaded, 0..1_000_000);
        assert!(published(&overloaded, vec![builder], &keys(0..10)).is_none());
    }

    #[test]
    fn budget_and_bloom_are_shared_by_all_builders() {
        let spec = spec(1000, 1000);
        let mut a = built(&spec, 0..600);
        let b = built(&spec, 600..1200);
        assert!(is_buffered(&a));
        assert!(!is_buffered(&b));
        // `a` goes over on its next batch, however small, into the same bloom
        // filter.
        a.extend(&keys(1200..1201)).unwrap();
        assert!(Arc::ptr_eq(
            planned_bloom(&a).unwrap(),
            planned_bloom(&b).unwrap()
        ));
    }

    #[test]
    fn skew_under_the_budget_stays_buffered() {
        let probe = keys(0..20_000);
        let random_state = PlRandomState::default();
        let build = |splits: &[std::ops::Range<i64>]| {
            let spec = spec_with_hashes(10, 100_000, random_state.clone());
            let builders = splits
                .iter()
                .map(|range| built(&spec, range.clone()))
                .collect::<Vec<_>>();
            assert!(builders.iter().all(is_buffered));
            published(&spec, builders, &probe).unwrap()
        };
        let skewed = build(&[0..90_000, 90_000..95_000, 95_000..100_000]);
        let even = build(&[0..33_000, 33_000..66_000, 66_000..100_000]);
        assert_eq!(skewed, even);
    }

    #[test]
    fn split_over_builders_does_not_matter() {
        let probe = keys(0..40_000);
        let random_state = PlRandomState::default();
        let spec = || spec_with_hashes(1000, 5_000, random_state.clone());

        // All keys in one builder, which goes over the budget halfway.
        let one = spec();
        let mut builder = built(&one, 0..4_000);
        builder.extend(&keys(4_000..8_000)).unwrap();
        let expected = published(&one, vec![builder], &probe).unwrap();
        assert!(expected.1[..8_000].iter().all(|m| *m));

        // Two builders, one of which stays buffered, in either order.
        for buffered_first in [true, false] {
            let two = spec();
            let buffered = built(&two, 0..4_000);
            let planned = built(&two, 4_000..8_000);
            assert!(is_buffered(&buffered));
            assert!(!is_buffered(&planned));
            let builders = if buffered_first {
                vec![buffered, planned]
            } else {
                vec![planned, buffered]
            };
            assert_eq!(expected, published(&two, builders, &probe).unwrap());
        }
    }

    /// A spec for integer keys of `dtype`, which can get a bitset.
    fn int_spec(dtype: DataType, rows_left: usize) -> KeyFilterSpec {
        KeyFilterSpec {
            dtype,
            int_keys: true,
            ..spec(1000, rows_left)
        }
    }

    fn mask(filter: &KeyFilter, probe: &Column) -> Vec<bool> {
        let mask = filter
            .evaluate(std::slice::from_ref(probe))
            .unwrap()
            .unwrap();
        mask.bool().unwrap().iter().map(|m| m.unwrap()).collect()
    }

    #[test]
    fn dense_keys_publish_the_range_only() {
        let spec = int_spec(DataType::Int64, BUFFERED_ROWS_BUDGET);
        let filter = KeyFilter::from_builders(&spec, vec![built(&spec, 0..1000)]);
        assert!(filter.bloom.is_none() && filter.bitset.is_none());
        let mask = mask(&filter, &keys(-5..1005));
        assert!(
            mask.iter()
                .enumerate()
                .all(|(i, m)| *m == (5..1005).contains(&i))
        );
    }

    #[test]
    fn sparse_keys_in_a_small_range_get_a_bitset() {
        let spec = int_spec(DataType::Int32, BUFFERED_ROWS_BUDGET);
        let build = |values: Vec<Option<i32>>| {
            let mut builder = KeyFilterBuilder::new(&spec);
            builder.extend(&Column::new("k".into(), values)).unwrap();
            builder
        };
        let builders = vec![
            build((0..500).map(|v| Some(v * 3)).collect()),
            build((500..1000).map(|v| Some(v * 3)).chain([None]).collect()),
        ];
        let filter = KeyFilter::from_builders(&spec, builders);
        assert!(filter.bloom.is_none() && filter.bitset.is_some());

        let probe: Vec<Option<i32>> = (-10..3010).map(Some).chain([None]).collect();
        let expected: Vec<bool> = probe
            .iter()
            .map(|v| v.is_some_and(|v| v % 3 == 0 && (0..3000).contains(&v)))
            .collect();
        assert_eq!(mask(&filter, &Column::new("k".into(), probe)), expected);
        // Another dtype is left alone.
        assert!(filter.evaluate(&[keys(0..10)]).unwrap().is_none());
    }

    #[test]
    fn wide_or_large_builds_keep_the_bloom_filter() {
        let spread = |range: std::ops::Range<i64>| {
            Column::new("k".into(), range.map(|v| v << 30).collect::<Vec<_>>())
        };
        let spec = int_spec(DataType::Int64, BUFFERED_ROWS_BUDGET);
        let mut builder = KeyFilterBuilder::new(&spec);
        builder.extend(&spread(0..1000)).unwrap();
        let filter = KeyFilter::from_builders(&spec, vec![builder]);
        assert!(filter.bloom.is_some() && filter.bitset.is_none());
        assert!(mask(&filter, &spread(0..1000)).iter().all(|m| *m));

        // Past the budget the keys are not kept, so even dense keys get a bloom.
        let spec = int_spec(DataType::Int64, 100);
        let filter = KeyFilter::from_builders(&spec, vec![built(&spec, 0..1000)]);
        assert!(filter.bloom.is_some() && filter.bitset.is_none());
    }

    #[test]
    fn no_builders_publish_an_empty_filter() {
        let spec = spec(1000, BUFFERED_ROWS_BUDGET);
        let (_, mask) = published(&spec, Vec::new(), &keys(0..1000)).unwrap();
        assert!(mask.iter().all(|m| !*m));
    }
}
