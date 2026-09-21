//! What a hash join knows about a build-side key, published to the scans
//! below its probe side once the build is done: the key's range, which scans
//! use to skip batches by their statistics, and optionally a bloom filter over
//! the keys, which scans probe per row.

use std::sync::Arc;

use polars_arrow::bitmap::BitmapBuilder;
use polars_core::config;
use polars_core::prelude::*;
use polars_expr::hash_keys::HashKeys;
use polars_io::predicates::{RuntimeRange, cast_bound};
use polars_plan::plans::options::{MAX_BUILD_PROBE_DISTINCT_RATIO, RuntimeFilter};
use polars_plan::plans::{PredicateExpr, TrivialPredicateExpr};
use polars_utils::bloom_filter::SplitBlockBloom;
use polars_utils::cardinality_sketch::CardinalitySketch;

/// Bits per key a bloom filter is sized for.
const BLOOM_BITS_PER_KEY: usize = 8;
/// A bloom filter with fewer bits per distinct key than this is not published.
const BLOOM_MIN_BITS_PER_KEY: usize = 4;
/// Smallest bloom filter that is built.
const BLOOM_MIN_BYTES: usize = 64 << 10;
/// Largest bloom filter that is published.
const BLOOM_MAX_BYTES: usize = 32 << 20;

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
}

impl KeyFilterSpec {
    fn new(filter: &RuntimeFilter, key_schema: &Schema) -> Self {
        Self {
            dtype: key_schema.get_at_index(filter.key_idx).unwrap().1.clone(),
            bloom_keys: filter.bloom_keys,
            probe_distinct: filter.probe_distinct,
            random_state: PlRandomState::default(),
        }
    }

    fn bloom(&self) -> Option<SplitBlockBloom> {
        let keys = self
            .bloom_keys?
            .max(BLOOM_MIN_BYTES * 8 / BLOOM_BITS_PER_KEY);
        let bytes = SplitBlockBloom::size_for(keys, BLOOM_BITS_PER_KEY);
        (bytes <= BLOOM_MAX_BYTES).then(|| SplitBlockBloom::with_capacity(keys, BLOOM_BITS_PER_KEY))
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
    bloom: SplitBlockBloom,
    sketch: CardinalitySketch,
}

impl KeyFilterBuilder {
    fn new(spec: &KeyFilterSpec) -> Self {
        Self {
            range: KeyRange::default(),
            bloom: spec.bloom().map(|bloom| BloomBuilder {
                spec: spec.clone(),
                bloom,
                sketch: CardinalitySketch::new(),
            }),
        }
    }

    /// Add the non-null values of `column`.
    fn extend(&mut self, column: &Column) -> PolarsResult<()> {
        self.range.extend(column)?;
        if let Some(b) = &mut self.bloom {
            b.spec.hash_keys(column).for_each_hash(|_, hash| {
                if let Some(hash) = hash {
                    b.bloom.insert(hash);
                    b.sketch.insert(hash);
                }
            });
        }
        Ok(())
    }

    /// Add everything `other` collected.
    pub(super) fn merge(&mut self, other: Self) {
        self.range.merge(other.range);
        if let (Some(a), Some(b)) = (&mut self.bloom, other.bloom) {
            a.bloom.union_with(&b.bloom);
            a.sketch.combine(&b.sketch);
        }
    }

    /// The filter to publish. The bloom filter is left out when it holds too
    /// many distinct keys for its size, or too large a share of the probe's
    /// distinct keys to be worth probing.
    fn finish(self) -> KeyFilter {
        let bloom = self.bloom.and_then(|b| {
            let distinct = b.sketch.estimate();
            let overloaded = distinct.saturating_mul(BLOOM_MIN_BITS_PER_KEY) > b.bloom.num_bits();
            let weak = b
                .spec
                .probe_distinct
                .is_some_and(|probe| distinct as f64 > probe as f64 * MAX_BUILD_PROBE_DISTINCT_RATIO);
            if (overloaded || weak) && config::verbose() {
                eprintln!(
                    "dropping bloom filter of {} bytes: {distinct} distinct build keys, {:?} distinct probe keys",
                    b.bloom.size_bytes(),
                    b.spec.probe_distinct
                );
            }
            (!overloaded && !weak).then_some(KeyBloom {
                spec: b.spec,
                bloom: b.bloom,
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
