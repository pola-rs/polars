//! Row-group pruning using Parquet split-block bloom filters.
//!
//! Called from [`super::statistics::calculate_row_group_pred_pushdown_skip_mask`].
//!
//! Statistics (min/max) run first; blooms are probed only on row groups not already
//! skipped by stats. The bloom mask is OR-merged with the statistics mask (set bit = skip).

use std::ops::Range;
use std::sync::Arc;

use arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_core::prelude::{PlHashMap, Scalar};
use polars_error::PolarsResult;
use polars_io::predicates::{
    ScanIOPredicate, SpecializedColumnPredicate, any_hashes_might_be_in_bloom_filter_bytes,
    bloom_hashes_for_scalars,
};
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use polars_parquet::read::{ColumnChunkMetadata, RowGroupMetadata};
use polars_utils::pl_str::PlSmallStr;

use super::projection::ArrowFieldProjection;

/// Maximum number of `is_in` literals to probe against a bloom filter.
///
/// Probing blooms is cheap for `col == literal`, but can become expensive for large `is_in` lists
/// because it scales with `(#row_groups × #values)`. Spark applies a similar threshold for Parquet
/// `IN` pushdown (`spark.sql.parquet.pushdown.inFilterThreshold`, default 10).
fn bloom_in_filter_threshold() -> usize {
    const DEFAULT: usize = 10;
    std::env::var("POLARS_BLOOM_IN_FILTER_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT)
}

/// Bloom-eligible column: on-disk Arrow field name and precomputed literal hashes.
pub(super) struct BloomColumnPred {
    pub(super) arrow_field_name: PlSmallStr,
    pub(super) hashes: Box<[u64]>,
}

/// Combine statistics- and bloom-derived skip masks.
///
/// Either path may mark a row group for skipping; we take the union (`|`).
pub(super) fn merge_row_group_skip_masks(
    statistics_mask: Bitmap,
    bloom_mask: Option<Bitmap>,
) -> Bitmap {
    let Some(bloom_mask) = bloom_mask else {
        return statistics_mask;
    };
    debug_assert_eq!(statistics_mask.len(), bloom_mask.len());
    &statistics_mask | &bloom_mask
}

/// Collect bloom-eligible columns and hash literals once per file.
pub(super) fn collect_bloom_preds(
    predicate: &ScanIOPredicate,
    projected_arrow_fields: &[ArrowFieldProjection],
) -> Option<Vec<BloomColumnPred>> {
    // Scan projection name → Arrow field name (bloom offsets live in chunk metadata by Arrow name).
    let projection_by_output: PlHashMap<_, _> = projected_arrow_fields
        .iter()
        .map(|p| (p.output_name().clone(), p.arrow_field().name.clone()))
        .collect();

    let mut bloom_preds = Vec::new();
    for (output_name, (_, specialized)) in predicate.column_predicates.predicates.iter() {
        if !predicate.live_columns.contains(output_name) {
            continue;
        }
        let Some(values) = bloom_pred_values(specialized) else {
            continue;
        };
        let Some(hashes) = bloom_hashes_for_scalars(values) else {
            continue;
        };
        let Some(arrow_field_name) = projection_by_output.get(output_name) else {
            continue;
        };
        bloom_preds.push(BloomColumnPred {
            arrow_field_name: arrow_field_name.clone(),
            hashes,
        });
    }

    (!bloom_preds.is_empty()).then_some(bloom_preds)
}

/// For each row group not already skipped by `statistics_mask`, probe on-disk bloom filters.
///
/// Returns `None` if there are no bloom predicates. Otherwise returns a bitmap of length
/// `row_groups.len()` where `true` means skip (value cannot be in bloom).
pub(super) async fn calculate_bloom_filter_skip_mask(
    row_groups: &[RowGroupMetadata],
    byte_source: Arc<DynByteSource>,
    bloom_preds: Option<Vec<BloomColumnPred>>,
    statistics_mask: &Bitmap,
) -> PolarsResult<Option<Bitmap>> {
    let Some(bloom_preds) = bloom_preds
        .as_deref()
        .filter(|p| !p.is_empty())
    else {
        return Ok(None);
    };

    debug_assert_eq!(statistics_mask.len(), row_groups.len());

    let mut skip = BitmapBuilder::with_capacity(row_groups.len());
    let mut bitset = Vec::new();

    for (i, rg) in row_groups.iter().enumerate() {
        if statistics_mask.get_bit(i) {
            // Already skipped by min/max; bloom probe would not change the merged mask.
            skip.push(false);
            continue;
        }
        skip.push(
            should_skip_row_group(rg, &bloom_preds, &byte_source, &mut bitset)
                .await?,
        );
    }

    Ok(Some(skip.freeze()))
}

/// Literals to hash into the bloom filter; `None` for non-point predicates (ranges, strings, …).
fn bloom_pred_values(specialized: &Option<SpecializedColumnPredicate>) -> Option<&[Scalar]> {
    match specialized {
        Some(SpecializedColumnPredicate::Equal(s)) => Some(std::slice::from_ref(s)),
        Some(SpecializedColumnPredicate::EqualOneOf(v)) => {
            (v.len() <= bloom_in_filter_threshold()).then_some(v.as_ref())
        },
        _ => None,
    }
}

/// Byte range of the serialized bloom filter for a column chunk, if present and valid.
///
/// Returns `None` on missing metadata or values we cannot read safely (caller treats as “might contain”).
fn bloom_byte_range(meta: &ColumnChunkMetadata) -> Option<Range<usize>> {
    let offset = meta.bloom_filter_offset()?;
    if offset < 0 {
        return None;
    }
    let offset = offset as usize;
    let len = meta.bloom_filter_length()?;
    if len <= 0 {
        return None;
    }
    let end = offset.checked_add(len as usize)?;
    Some(offset..end)
}

/// Returns `true` if blooms prove this row group cannot satisfy the filter conjuncts.
async fn should_skip_row_group(
    rg: &RowGroupMetadata,
    bloom_preds: &[BloomColumnPred],
    byte_source: &DynByteSource,
    bitset: &mut Vec<u8>,
) -> PolarsResult<bool> {
    for pred in bloom_preds {
        let Some(idxs) = rg.columns_idxs_under_root_iter(pred.arrow_field_name.as_str()) else {
            continue;
        };
        // Structs/lists map to 0 or 2+ chunks; blooms are per chunk, so we cannot pick a single bloom for a nested field.
        if idxs.len() != 1 {
            continue;
        }
        let column_metadata = &rg.parquet_columns()[idxs[0]];
        let Some(range) = bloom_byte_range(column_metadata) else {
            continue;
        };

        let bloom_bytes = byte_source.get_range(range).await?;

        let any_might_match =
            any_hashes_might_be_in_bloom_filter_bytes(&pred.hashes, bloom_bytes.as_ref(), bitset)
                .unwrap_or(true);

        if !any_might_match {
            return Ok(true);
        }
    }
    Ok(false)
}
