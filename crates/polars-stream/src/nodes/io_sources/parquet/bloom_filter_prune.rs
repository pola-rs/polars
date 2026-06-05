//! Row-group pruning using Parquet split-block bloom filters.
//!
//! Called from [`super::statistics::calculate_row_group_pred_pushdown_skip_mask`].
//!
//! Statistics (min/max) run first; blooms are probed only on row groups not already
//! skipped by stats. The bloom mask is OR-merged with the statistics mask (set bit = skip).
//!
//! Blooms are probabilistic; a fully dictionary-encoded chunk's dictionary page gives exact
//! membership and should take precedence once dictionary-based skipping exists (cost-gated by
//! dictionary size). See the design note on [`super::statistics::calculate_row_group_pred_pushdown_skip_mask`].

use std::ops::Range;
use std::sync::Arc;

use arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_core::prelude::{PlHashMap, Scalar};
use polars_error::PolarsResult;
use polars_io::predicates::{
    ScanIOPredicate, SpecializedColumnPredicate, bloom_hashes_for_scalars,
};
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use polars_parquet::parquet::bloom_filter::{
    BLOCK_SIZE, any_hashes_might_be_in_blocks, bloom_filter_layout, might_contain_any_hashes,
    prefer_block_reads, unique_block_indices,
};
use polars_parquet::read::{ColumnChunkMetadata, RowGroupMetadata};
use polars_utils::pl_str::PlSmallStr;

use super::projection::ArrowFieldProjection;

/// Column to probe with a bloom: on-disk Arrow field name and precomputed literal hashes.
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
    // Name translation table from what the filter uses to what Parquet stores on disk (bloom offsets live in chunk metadata by Arrow name).
    let output_to_arrow_name: PlHashMap<_, _> = projected_arrow_fields
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
        let Some(arrow_field_name) = output_to_arrow_name.get(output_name) else {
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
pub(super) async fn bloom_filter_row_group_skip_mask(
    row_groups: &[RowGroupMetadata],
    byte_source: Arc<DynByteSource>,
    bloom_preds: Option<Vec<BloomColumnPred>>,
    statistics_mask: &Bitmap,
) -> PolarsResult<Option<Bitmap>> {
    let Some(bloom_preds) = bloom_preds.as_deref().filter(|p| !p.is_empty()) else {
        return Ok(None);
    };

    debug_assert_eq!(statistics_mask.len(), row_groups.len());

    let mut skip = BitmapBuilder::with_capacity(row_groups.len());
    let mut bitset = Vec::new();

    for (i, rg) in row_groups.iter().enumerate() {
        if statistics_mask.get_bit(i) {
            // Already skipped by min/max; bloom probe would not change the merged mask
            // so this value is irrelevant.
            skip.push(false);
            continue;
        }
        skip.push(should_skip_row_group(rg, bloom_preds, &byte_source, &mut bitset).await?);
    }

    Ok(Some(skip.freeze()))
}

/// Literals to hash into the bloom filter; `None` for non-point predicates (ranges, strings, …).
fn bloom_pred_values(specialized: &Option<SpecializedColumnPredicate>) -> Option<&[Scalar]> {
    match specialized {
        Some(SpecializedColumnPredicate::Equal(s)) => Some(std::slice::from_ref(s)),
        Some(SpecializedColumnPredicate::EqualOneOf(v)) => {
            (v.len() <= polars_config::config().bloom_in_filter_threshold()).then_some(v.as_ref())
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

        let any_might_match = probe_bloom_hashes(&pred.hashes, range, byte_source, bitset)
            .await
            .unwrap_or(true);

        if !any_might_match {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Max bytes to read for the Thrift bloom filter header; Apache Arrow seems to think 20 is enough.
const BLOOM_HEADER_READ_CAP: usize = 64;

/// Probe bloom filter literals, reading only the split-block(s) needed when cheaper than the full slice.
async fn probe_bloom_hashes(
    hashes: &[u64],
    bloom_range: Range<usize>,
    byte_source: &DynByteSource,
    bitset: &mut Vec<u8>,
) -> PolarsResult<bool> {
    let header_end = bloom_range
        .end
        .min(bloom_range.start.saturating_add(BLOOM_HEADER_READ_CAP));
    if header_end <= bloom_range.start {
        return Ok(true);
    }

    let prefix = byte_source.get_range(bloom_range.start..header_end).await?;
    // Unsupported, truncated, or corrupt header: treat as inconclusive (may contain matches).
    let Some(layout) = bloom_filter_layout(prefix.as_ref()).ok().flatten() else {
        return Ok(true);
    };

    let bitset_start = bloom_range.start + layout.header_len;
    let bitset_end = bitset_start
        .checked_add(layout.bitset_num_bytes)
        .filter(|&end| end <= bloom_range.end);
    if bitset_end.is_none() {
        return Ok(true);
    }

    let block_indices = unique_block_indices(hashes, layout.bitset_num_bytes);
    if !prefer_block_reads(block_indices.len(), &layout, bloom_range.len()) {
        let bloom_bytes = byte_source.get_range(bloom_range).await?;
        // Corrupt or truncated bloom bytes: inconclusive, do not skip the row group.
        return Ok(might_contain_any_hashes(bloom_bytes.as_ref(), hashes, bitset).unwrap_or(true));
    }

    let mut block_ranges: Vec<Range<usize>> = block_indices
        .iter()
        .map(|&idx| {
            let start = bitset_start + idx * BLOCK_SIZE;
            start..start + BLOCK_SIZE
        })
        .collect();
    let blocks_by_offset = byte_source.get_ranges(&mut block_ranges).await?;

    Ok(any_hashes_might_be_in_blocks(
        hashes,
        layout.bitset_num_bytes,
        |idx| {
            let start = bitset_start + idx * BLOCK_SIZE;
            blocks_by_offset.get(&start).map(|b| b.as_ref())
        },
    ))
}
