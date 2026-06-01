//! Row-group pruning using Parquet split-block bloom filters.
//!
//! Called from [`super::statistics::calculate_row_group_pred_pushdown_skip_mask`].
//! Produces a per–row-group skip mask that is OR-merged with the statistics mask:
//! a set bit means "skip this row group" (do not decode it).

use std::ops::Range;
use std::sync::Arc;

use arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_core::prelude::{PlHashMap, Scalar};
use polars_error::PolarsResult;
use polars_io::predicates::{
    ScanIOPredicate, SpecializedColumnPredicate, any_scalar_might_be_in_bloom_filter_bytes,
};
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use polars_parquet::read::{ColumnChunkMetadata, RowGroupMetadata};
use polars_utils::pl_str::PlSmallStr;

use super::projection::ArrowFieldProjection;

/// On-disk Arrow field name and equality literals to probe.
type BloomPred = (PlSmallStr, Box<[Scalar]>);

/// Combine statistics- and bloom-derived skip masks.
///
/// Either path may mark a row group for skipping; we take the union (`|`).
pub(super) fn merge_row_group_skip_masks(statistics_mask: Bitmap, bloom_mask: Option<Bitmap>) -> Bitmap {
    let Some(bloom_mask) = bloom_mask else {
        return statistics_mask;
    };
    debug_assert_eq!(statistics_mask.len(), bloom_mask.len());
    &statistics_mask | &bloom_mask
}

/// Whether the scan predicate has equality / `is_in` literals that could use blooms.
pub(super) fn has_bloom_eligible_predicates(
    predicate: &ScanIOPredicate,
    projected_arrow_fields: &[ArrowFieldProjection],
) -> bool {
    collect_bloom_preds(predicate, projected_arrow_fields).is_some()
}

/// For each row group, probe on-disk bloom filters for equality / `is_in` literals.
///
/// Returns `None` if there are no bloom-eligible predicates. Otherwise returns a bitmap
/// of length `row_groups.len()` where `true` means skip (value cannot be in bloom).
pub(super) async fn calculate_bloom_filter_skip_mask(
    row_groups: &[RowGroupMetadata],
    byte_source: Arc<DynByteSource>,
    predicate: &ScanIOPredicate,
    projected_arrow_fields: &[ArrowFieldProjection],
) -> PolarsResult<Option<Bitmap>> {
    let Some(bloom_preds) = collect_bloom_preds(predicate, projected_arrow_fields) else {
        return Ok(None);
    };

    let mut skip = BitmapBuilder::with_capacity(row_groups.len());
    let mut bitset = Vec::new();

    for rg in row_groups {
        skip.push(
            should_skip_row_group(rg, &bloom_preds, &byte_source, &mut bitset).await?,
        );
    }

    Ok(Some(skip.freeze()))
}

/// Extract per-column literals to probe, keyed by on-disk Arrow field name.
///
/// Uses [`SpecializedColumnPredicate`] (`Equal` / `EqualOneOf` only). Returns `None` if
/// nothing qualifies. Shared with [`has_bloom_eligible_predicates`] for the statistics early exit.
fn collect_bloom_preds(
    predicate: &ScanIOPredicate,
    projected_arrow_fields: &[ArrowFieldProjection],
) -> Option<Vec<BloomPred>> {
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
        let values = bloom_pred_values(specialized)?;
        let arrow_field_name = projection_by_output.get(output_name)?;
        bloom_preds.push((arrow_field_name.clone(), values));
    }

    (!bloom_preds.is_empty()).then_some(bloom_preds)
}

/// Literals to hash into the bloom filter; `None` for non-point predicates (ranges, strings, …).
fn bloom_pred_values(specialized: &Option<SpecializedColumnPredicate>) -> Option<Box<[Scalar]>> {
    match specialized {
        Some(SpecializedColumnPredicate::Equal(s)) => Some(Box::new([s.clone()])),
        Some(SpecializedColumnPredicate::EqualOneOf(v)) => Some(v.clone()),
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
    bloom_preds: &[BloomPred],
    byte_source: &DynByteSource,
    bitset: &mut Vec<u8>,
) -> PolarsResult<bool> {
    for (arrow_field_name, values) in bloom_preds {
        let Some(idxs) = rg.columns_idxs_under_root_iter(arrow_field_name.as_str()) else {
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

        let any_might_match = any_scalar_might_be_in_bloom_filter_bytes(values, bloom_bytes.as_ref(), bitset)
            .unwrap_or(true);

        if !any_might_match {
            return Ok(true);
        }
    }
    Ok(false)
}
