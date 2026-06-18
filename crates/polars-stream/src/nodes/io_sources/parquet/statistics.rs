use std::ops::Range;

use arrow::array::{Array, MutablePrimitiveArray, PrimitiveArray, StructArray};
use arrow::bitmap::Bitmap;
use arrow::pushable::Pushable;
use polars_async::executor::{self, TaskPriority};
use polars_core::prelude::*;
use polars_io::RowIndex;
use polars_io::predicates::ScanIOPredicate;
use polars_io::prelude::FileMetadata;
use polars_io::utils::byte_source::DynByteSource;
use polars_parquet::read::RowGroupMetadata;
use polars_parquet::read::statistics::{ArrowColumnStatisticsArrays, deserialize_all};
use polars_plan::plans::predicates::null_count_dtype;
use polars_utils::format_pl_smallstr;

use crate::nodes::io_sources::parquet::bloom_filter_prune::{
    bloom_filter_row_group_skip_mask, collect_bloom_preds, merge_row_group_skip_masks,
};
use crate::nodes::io_sources::parquet::projection::ArrowFieldProjection;

/// Arguments for [`calculate_row_group_pred_pushdown_skip_mask`].
///
/// Stored in a struct to avoid tripping clippy's `too_many_arguments` lint.
pub(super) struct RowGroupPredPushdownArgs<'a> {
    pub row_group_slice: Range<usize>,
    pub use_statistics: bool,
    pub predicate: Option<&'a ScanIOPredicate>,
    pub metadata: &'a Arc<FileMetadata>,
    pub projected_arrow_fields: Arc<[ArrowFieldProjection]>,
    /// File bytes for bloom filters (not stored in `metadata.footer_buf`).
    pub byte_source: Arc<DynByteSource>,
    /// Updated to the position of the first row group.
    pub row_index: Option<RowIndex>,
    pub verbose: bool,
}

struct StatisticsColumns {
    min: Column,
    max: Column,
    null_count: Column,
}

impl StatisticsColumns {
    fn new_null(dtype: &DataType, height: usize) -> Self {
        Self {
            min: Column::full_null(PlSmallStr::EMPTY, height, dtype),
            max: Column::full_null(PlSmallStr::EMPTY, height, dtype),
            null_count: Column::full_null(PlSmallStr::EMPTY, height, &null_count_dtype(dtype)),
        }
    }

    fn from_arrow_statistics(
        statistics: ArrowColumnStatisticsArrays,
        field: &ArrowField,
    ) -> PolarsResult<Self> {
        Ok(Self {
            min: unsafe {
                Series::_try_from_arrow_unchecked_with_md(
                    PlSmallStr::EMPTY,
                    vec![statistics.min_value],
                    field.dtype(),
                    field.metadata.as_deref(),
                )
            }?
            .into_column(),

            max: unsafe {
                Series::_try_from_arrow_unchecked_with_md(
                    PlSmallStr::EMPTY,
                    vec![statistics.max_value],
                    field.dtype(),
                    field.metadata.as_deref(),
                )
            }?
            .into_column(),

            null_count: Series::from_arrow(PlSmallStr::EMPTY, statistics.null_count.boxed())?
                .into_column(),
        })
    }

    fn with_base_column_name(self, base_column_name: &str) -> Self {
        let b = base_column_name;

        let min = self.min.with_name(format_pl_smallstr!("{b}_min"));
        let max = self.max.with_name(format_pl_smallstr!("{b}_max"));
        let null_count = self.null_count.with_name(format_pl_smallstr!("{b}_nc"));

        Self {
            min,
            max,
            null_count,
        }
    }
}

/// Builds a per–row-group skip mask from predicate pushdown (set bit = skip row group).
///
/// Two mechanisms are merged with bitwise OR (stats first, then blooms on survivors):
/// - **Statistics** (`skip_batch_predicate` over min/max/null_count from the footer).
/// - **Bloom filters** (equality / `is_in` literals probed via `byte_source` range reads).
///
/// **Future — dictionary pruning:** On a fully dictionary-encoded column chunk, the dictionary
/// page is an exact membership set (no false positives), strictly stronger than a bloom. Prefer
/// dictionary probing on such chunks and fall back to a bloom filter when one is present.
/// Probing the dictionary means decoding it and building a lookup set (cost scales with
/// dictionary size), so dictionary pruning should be cost-based — enabled when the dictionary
/// is small. Not implemented yet.
pub(super) async fn calculate_row_group_pred_pushdown_skip_mask(
    args: RowGroupPredPushdownArgs<'_>,
) -> PolarsResult<Option<Bitmap>> {
    let RowGroupPredPushdownArgs {
        row_group_slice,
        use_statistics,
        predicate,
        metadata,
        projected_arrow_fields,
        byte_source,
        mut row_index,
        verbose,
    } = args;

    let Some(predicate) = predicate else {
        return Ok(None);
    };

    // `use_statistics` gates Parquet *column statistics* (min/max/null_count) only.
    // Bloom filters are separate footer metadata and are not disabled by this flag.
    let has_skip_batch_predicate = use_statistics && predicate.skip_batch_predicate.is_some();
    let bloom_preds = if polars_config::config().bloom_filter_prune() {
        collect_bloom_preds(predicate, projected_arrow_fields.as_ref())
    } else {
        None
    };
    let has_bloom_predicates = bloom_preds.is_some();

    if !has_skip_batch_predicate && !has_bloom_predicates {
        return Ok(None);
    }

    // Clone the skip batch predicate for the spawned task.
    let sbp = predicate.skip_batch_predicate.clone();

    let num_row_groups = row_group_slice.len();
    let metadata = metadata.clone();
    let live_columns = predicate.live_columns.clone();
    let projected_arrow_fields = projected_arrow_fields.clone();
    // Cloned into the spawned task for bloom `get_range` calls (same `Arc` as row-group fetch).
    let byte_source = byte_source.clone();

    // Note: We are spawning here onto the computational async runtime because the caller is being run
    // on a tokio async thread.
    let skip_row_group_mask = executor::spawn(TaskPriority::High, async move {
        let row_groups_slice = &metadata.row_groups[row_group_slice.clone()];

        if let Some(ri) = &mut row_index {
            for md in metadata.row_groups[0..row_group_slice.start].iter() {
                ri.offset = ri
                    .offset
                    .saturating_add(IdxSize::try_from(md.num_rows()).unwrap_or(IdxSize::MAX));
            }
        }

        let statistics_mask = match sbp {
            Some(sbp) if has_skip_batch_predicate => {
                let mut columns = Vec::with_capacity(1 + live_columns.len() * 3);

                let lengths: Vec<IdxSize> = row_groups_slice
                    .iter()
                    .map(|rg| rg.num_rows() as IdxSize)
                    .collect();

                columns.push(Column::new("len".into(), lengths));

                for projection in projected_arrow_fields.iter() {
                    let c = projection.output_name();

                    if !live_columns.contains(c) {
                        continue;
                    }

                    let mut statistics = load_parquet_column_statistics(
                        row_groups_slice,
                        projection,
                        &metadata.footer_buf,
                    )?;

                    // Note: Order is important here. We re-use the transform for the output column, meaning
                    // that it may set the column name.
                    statistics.min = projection.apply_transform(statistics.min)?;
                    statistics.max = projection.apply_transform(statistics.max)?;

                    let statistics = statistics.with_base_column_name(c);

                    columns.extend([statistics.min, statistics.max, statistics.null_count]);
                }

                if let Some(row_index) = row_index {
                    let statistics = build_row_index_statistics(&row_index, row_groups_slice)
                        .with_base_column_name(&row_index.name);

                    columns.extend([statistics.min, statistics.max, statistics.null_count]);
                }

                let statistics_df = DataFrame::new(num_row_groups, columns)?;
                sbp.evaluate_with_stat_df(&statistics_df)?
            },
            _ => {
                // Stats disabled or no skip_batch_predicate: do not prune from min/max/null_count.
                Bitmap::new_with_value(false, num_row_groups)
            },
        };

        let bloom_mask = bloom_filter_row_group_skip_mask(
            row_groups_slice,
            byte_source,
            bloom_preds,
            &statistics_mask,
        )
        .await?;

        // Skip if either stats or bloom proved the row group cannot match.
        PolarsResult::Ok(merge_row_group_skip_masks(statistics_mask, bloom_mask))
    })
    .await?;

    if verbose {
        eprintln!(
            "[ParquetFileReader]: Predicate pushdown: \
            reading {} / {} row groups",
            skip_row_group_mask.unset_bits(),
            num_row_groups,
        );
    }

    Ok(Some(skip_row_group_mask))
}

/// Assembled `min` / `max` / `null_count` statistics arrays for a (possibly nested) struct field.
struct StructStatisticsArrays {
    min: Box<dyn Array>,
    max: Box<dyn Array>,
    null_count: Box<dyn Array>,
}

/// Recursively assemble per-field `min` / `max` / `null_count` statistics arrays for a
/// (possibly nested) struct field, consuming one parquet leaf column per scalar leaf. Returns
/// `None` (signalling the caller to fall back to null statistics) for an empty struct, an
/// unsupported leaf type (e.g. a nested list), or if the leaves run out before the fields do.
fn build_struct_statistics_arrays(
    field: &ArrowField,
    row_groups: &[RowGroupMetadata],
    leaf_idxs: &[usize],
    cursor: &mut usize,
    footer_buf: &[u8],
) -> PolarsResult<Option<StructStatisticsArrays>> {
    let height = row_groups.len();
    match field.dtype() {
        ArrowDataType::Struct(children) => {
            // An empty struct has no leaf statistics to reason about, and `StructArray::new`
            // panics on a struct dtype with no children, so bail to null statistics.
            if children.is_empty() {
                return Ok(None);
            }

            let mut mins = Vec::with_capacity(children.len());
            let mut maxs = Vec::with_capacity(children.len());
            let mut ncs = Vec::with_capacity(children.len());

            // Pairs each arrow struct field with the next parquet leaf by position. Both derive
            // from the same parquet schema, so their leaf orders match; the caller's `cursor`
            // length check catches a leaf-count mismatch.
            for child in children {
                let Some(child) = build_struct_statistics_arrays(
                    child, row_groups, leaf_idxs, cursor, footer_buf,
                )?
                else {
                    return Ok(None);
                };
                mins.push(child.min);
                maxs.push(child.max);
                ncs.push(child.null_count);
            }

            let min = StructArray::new(field.dtype().clone(), height, mins, None).to_boxed();
            let max = StructArray::new(field.dtype().clone(), height, maxs, None).to_boxed();

            // The null-count struct mirrors the field's shape with each leaf replaced by the
            // index type; read that shape straight off the assembled per-field arrays.
            let nc_fields: Vec<ArrowField> = children
                .iter()
                .zip(ncs.iter())
                .map(|(child, nc)| ArrowField::new(child.name.clone(), nc.dtype().clone(), true))
                .collect();
            let null_count =
                StructArray::new(ArrowDataType::Struct(nc_fields), height, ncs, None).to_boxed();

            Ok(Some(StructStatisticsArrays {
                min,
                max,
                null_count,
            }))
        },
        _ => {
            // Scalar leaf: consume the next parquet leaf column.
            if *cursor >= leaf_idxs.len() {
                return Ok(None);
            }
            let idx = leaf_idxs[*cursor];
            *cursor += 1;

            match deserialize_all(field, row_groups, idx, footer_buf)? {
                Some(statistics) => Ok(Some(StructStatisticsArrays {
                    min: statistics.min_value,
                    max: statistics.max_value,
                    null_count: statistics.null_count.to_boxed(),
                })),
                // Unsupported leaf type (e.g. a list nested inside the struct).
                None => Ok(None),
            }
        },
    }
}

fn load_struct_column_statistics(
    arrow_field: &ArrowField,
    row_groups: &[RowGroupMetadata],
    leaf_idxs: &[usize],
    footer_buf: &[u8],
) -> PolarsResult<Option<StatisticsColumns>> {
    let mut cursor = 0;
    let Some(StructStatisticsArrays {
        min,
        max,
        null_count,
    }) = build_struct_statistics_arrays(
        arrow_field,
        row_groups,
        leaf_idxs,
        &mut cursor,
        footer_buf,
    )?
    else {
        return Ok(None);
    };

    // Only trust the assembled stats if every parquet leaf mapped to a struct field; a
    // mismatch means the schema and parquet layout disagree and the stats could be misaligned.
    if cursor != leaf_idxs.len() {
        return Ok(None);
    }

    let min = unsafe {
        Series::_try_from_arrow_unchecked_with_md(
            PlSmallStr::EMPTY,
            vec![min],
            arrow_field.dtype(),
            arrow_field.metadata.as_deref(),
        )
    }?
    .into_column();
    let max = unsafe {
        Series::_try_from_arrow_unchecked_with_md(
            PlSmallStr::EMPTY,
            vec![max],
            arrow_field.dtype(),
            arrow_field.metadata.as_deref(),
        )
    }?
    .into_column();
    let null_count = Series::from_arrow(PlSmallStr::EMPTY, null_count)?.into_column();

    Ok(Some(StatisticsColumns {
        min,
        max,
        null_count,
    }))
}

fn load_parquet_column_statistics(
    row_groups: &[RowGroupMetadata],
    projection: &ArrowFieldProjection,
    footer_buf: &[u8],
) -> PolarsResult<StatisticsColumns> {
    let arrow_field = projection.arrow_field();

    let null_statistics = || {
        Ok(StatisticsColumns::new_null(
            &DataType::from_arrow_field(arrow_field),
            row_groups.len(),
        ))
    };

    // This can be None in the allow_missing_columns case.
    let Some(idxs) = row_groups[0].columns_idxs_under_root_iter(&arrow_field.name) else {
        return null_statistics();
    };

    // Structs span multiple parquet leaf columns; assemble per-field statistics so the
    // skip-batch predicate can prune on an individual struct field. Falls back to (struct-
    // shaped) null statistics if any leaf is unsupported.
    if matches!(arrow_field.dtype(), ArrowDataType::Struct(_)) {
        let Some(statistics) =
            load_struct_column_statistics(arrow_field, row_groups, idxs, footer_buf)?
        else {
            return null_statistics();
        };
        return Ok(statistics);
    }

    // Structs are handled above, so only non-struct columns reach here. A scalar occupies
    // exactly one leaf and is read below; multi-leaf nested types (lists, maps) don't have
    // statistics we read. The empty case is defensive: a present root always has >= 1 leaf.
    if idxs.is_empty() || idxs.len() > 1 {
        return null_statistics();
    }

    let idx = idxs[0];

    let Some(statistics) = deserialize_all(arrow_field, row_groups, idx, footer_buf)? else {
        return null_statistics();
    };

    StatisticsColumns::from_arrow_statistics(statistics, arrow_field)
}

fn build_row_index_statistics(
    row_index: &RowIndex,
    row_groups: &[RowGroupMetadata],
) -> StatisticsColumns {
    let mut offset = row_index.offset;

    let null_count = PrimitiveArray::<IdxSize>::full(row_groups.len(), 0, ArrowDataType::IDX_DTYPE);

    let mut min_value = MutablePrimitiveArray::<IdxSize>::with_capacity(row_groups.len());
    let mut max_value = MutablePrimitiveArray::<IdxSize>::with_capacity(row_groups.len());

    for rg in row_groups.iter() {
        let n_rows = IdxSize::try_from(rg.num_rows()).unwrap_or(IdxSize::MAX);

        if offset.checked_add(n_rows).is_none() {
            min_value.push_null();
            max_value.push_null();
            continue;
        }

        if n_rows == 0 {
            min_value.push_null();
            max_value.push_null();
        } else {
            min_value.push_value(offset);
            max_value.push_value(offset + n_rows - 1);
        }

        offset = offset.saturating_add(n_rows);
    }

    StatisticsColumns {
        min: Series::from_array(PlSmallStr::EMPTY, min_value.freeze()).into_column(),
        max: Series::from_array(PlSmallStr::EMPTY, max_value.freeze()).into_column(),
        null_count: Series::from_array(PlSmallStr::EMPTY, null_count).into_column(),
    }
}
