use std::cmp::Ordering;
use std::ops::Range;

use polars_arrow::array::{Array, MutablePrimitiveArray, PrimitiveArray, StructArray};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::pushable::Pushable;
use polars_async::executor::{self, TaskPriority};
use polars_core::prelude::*;
use polars_io::RowIndex;
use polars_io::predicates::{RuntimeRange, RuntimeRangeHint, ScanIOPredicate};
use polars_io::prelude::FileMetadata;
use polars_parquet::read::RowGroupMetadata;
use polars_parquet::read::statistics::{
    ArrowBound, ArrowColumnStatisticsArrays, BoundConversion, LeafBounds, deserialize_all,
};
use polars_plan::plans::predicates::null_count_dtype;
use polars_utils::format_pl_smallstr;

use crate::nodes::io_sources::parquet::projection::ArrowFieldProjection;

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

pub(super) async fn calculate_row_group_pred_pushdown_skip_mask(
    row_group_slice: Range<usize>,
    use_statistics: bool,
    predicate: Option<&ScanIOPredicate>,
    metadata: &Arc<FileMetadata>,
    projected_arrow_fields: Arc<[ArrowFieldProjection]>,
    row_index: Option<RowIndex>,
    verbose: bool,
) -> PolarsResult<Option<Bitmap>> {
    if !use_statistics {
        return Ok(None);
    }

    let Some(predicate) = predicate else {
        return Ok(None);
    };

    let static_mask = match predicate.skip_batch_predicate.as_ref() {
        Some(sbp) => {
            static_skip_mask(
                row_group_slice.clone(),
                sbp.clone(),
                predicate.live_columns.clone(),
                metadata,
                projected_arrow_fields.clone(),
                row_index,
            )
            .await?
        },
        None => None,
    };

    let mask = if predicate.runtime_ranges.is_empty() {
        static_mask
    } else {
        let runtime_mask = runtime_range_skip_mask(
            &predicate.runtime_ranges,
            metadata,
            row_group_slice.clone(),
            &projected_arrow_fields,
            static_mask.as_ref(),
        );
        match (static_mask, runtime_mask) {
            (Some(s), Some(r)) => Some(&s | &r),
            (s, r) => s.or(r),
        }
    };
    if verbose && (mask.is_some() || !predicate.runtime_ranges.is_empty()) {
        let num_row_groups = row_group_slice.len();
        eprintln!(
            "[ParquetFileReader]: Predicate pushdown: \
            reading {} / {} row groups",
            mask.as_ref().map_or(num_row_groups, |m| m.unset_bits()),
            num_row_groups,
        );
    }
    Ok(mask)
}

async fn static_skip_mask(
    row_group_slice: Range<usize>,
    sbp: Arc<dyn polars_io::predicates::SkipBatchPredicate>,
    skip_batch_columns: Arc<PlIndexSet<PlSmallStr>>,
    metadata: &Arc<FileMetadata>,
    projected_arrow_fields: Arc<[ArrowFieldProjection]>,
    mut row_index: Option<RowIndex>,
) -> PolarsResult<Option<Bitmap>> {
    let num_row_groups = row_group_slice.len();
    let metadata = metadata.clone();

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

        let mut columns = Vec::with_capacity(1 + skip_batch_columns.len() * 3);

        let lengths: Vec<IdxSize> = row_groups_slice
            .iter()
            .map(|rg| rg.num_rows() as IdxSize)
            .collect();

        columns.push(Column::new("len".into(), lengths));

        for projection in projected_arrow_fields.iter() {
            let c = projection.output_name();

            if !skip_batch_columns.contains(c) {
                continue;
            }

            let mut statistics =
                load_parquet_column_statistics(&metadata, row_group_slice.clone(), projection)?;

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

        sbp.evaluate_with_stat_df(&statistics_df)
    })
    .await?;

    Ok(Some(skip_row_group_mask))
}

/// A value of a column as polars stores it: a published range bound, or a row
/// group's bound. Integers of any width are one integer; bytes compare with
/// bytes only.
enum Bound<B> {
    Int(i128),
    Bytes(B),
}

impl<A: AsRef<[u8]>, B: AsRef<[u8]>> PartialEq<Bound<B>> for Bound<A> {
    fn eq(&self, other: &Bound<B>) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

impl<A: AsRef<[u8]>, B: AsRef<[u8]>> PartialOrd<Bound<B>> for Bound<A> {
    fn partial_cmp(&self, other: &Bound<B>) -> Option<Ordering> {
        match (self, other) {
            (Self::Int(a), Bound::Int(b)) => Some(a.cmp(b)),
            (Self::Bytes(a), Bound::Bytes(b)) => Some(a.as_ref().cmp(b.as_ref())),
            _ => None,
        }
    }
}

impl Bound<Vec<u8>> {
    /// A published range bound, or `None` when the published and the file's
    /// types cannot be compared exactly: every integer type compares with every
    /// other, any other type must match the file's.
    fn from_scalar(scalar: &Scalar, file_dtype: &DataType) -> Option<Self> {
        if scalar.dtype() != file_dtype && !(scalar.dtype().is_integer() && file_dtype.is_integer())
        {
            return None;
        }
        match scalar.value() {
            AnyValue::String(v) => Some(Self::Bytes(v.as_bytes().to_vec())),
            AnyValue::StringOwned(v) => Some(Self::Bytes(v.as_bytes().to_vec())),
            AnyValue::Binary(v) => Some(Self::Bytes(v.to_vec())),
            AnyValue::BinaryOwned(v) => Some(Self::Bytes(v.clone())),
            value => value.clone().to_physical().extract::<i128>().map(Self::Int),
        }
    }
}

impl<'a> Bound<&'a [u8]> {
    /// A row group's bound as polars stores the column, multiplied by `scale`
    /// like the values are, see [`DataType::arrow_value_scale`]. `None` when
    /// the multiplication wraps, as the values then have no order.
    fn from_arrow(bound: ArrowBound<'a>, scale: i64) -> Option<Self> {
        use ArrowBound as A;
        let int = match bound {
            A::Boolean(v) => v as i128,
            A::Int8(v) => v as i128,
            A::Int16(v) => v as i128,
            A::Int32(v) => v as i128,
            A::Int64(v) => v as i128,
            A::UInt8(v) => v as i128,
            A::UInt16(v) => v as i128,
            A::UInt32(v) => v as i128,
            A::UInt64(v) => v as i128,
            A::Int128(v) => v,
            A::Bytes(v) => return Some(Self::Bytes(v)),
            A::Str(v) => return Some(Self::Bytes(v.as_bytes())),
            A::Float16(_) | A::Float32(_) | A::Float64(_) | A::Int256(_) => return None,
        };
        let scaled = int * scale as i128;
        (scale == 1 || i64::try_from(scaled).is_ok()).then_some(Self::Int(scaled))
    }
}

/// Whether the values a leaf's bounds convert to compare exactly with a
/// published range.
fn compares_exactly(conversion: BoundConversion) -> bool {
    use BoundConversion as C;
    !matches!(
        conversion,
        C::Float16 | C::Float32 | C::Float64 | C::Decimal256
    )
}

/// Which row groups the runtime ranges skip, among those `static_mask` keeps.
/// `None` when they skip nothing. Every range is read once, before any bound is
/// decoded, and only the kept groups' bounds are decoded.
fn runtime_range_skip_mask(
    hints: &[RuntimeRangeHint],
    metadata: &FileMetadata,
    row_group_slice: Range<usize>,
    projected_arrow_fields: &[ArrowFieldProjection],
    static_mask: Option<&Bitmap>,
) -> Option<Bitmap> {
    let row_groups = &metadata.row_groups[row_group_slice.clone()];
    let mut mask: Option<MutableBitmap> = None;
    let is_kept = |mask: &Option<MutableBitmap>, i: usize| {
        !static_mask.is_some_and(|m| m.get_bit(i)) && !mask.as_ref().is_some_and(|m| m.get(i))
    };

    for hint in hints {
        if mask.as_ref().is_some_and(|m| m.unset_bits() == 0) {
            break;
        }
        let range = hint.source.runtime_range();
        let skip_all = match (&hint.constant, &range) {
            (Some(value), _) => RuntimeRangeHint::constant_matches(&range, value) == Some(false),
            (None, RuntimeRange::Pending | RuntimeRange::Disabled) => false,
            (None, RuntimeRange::Empty) => true,
            (None, RuntimeRange::Range { lo, hi }) => {
                let resolved = projected_arrow_fields
                    .iter()
                    .find(|p| p.output_name() == &hint.column)
                    .and_then(|projection| {
                        let arrow_field = projection.arrow_field();
                        let idxs = row_groups
                            .first()?
                            .columns_idxs_under_root_iter(&arrow_field.name)?;
                        let [idx] = idxs else { return None };
                        let leaf = LeafBounds::new(arrow_field, metadata, *idx)
                            .filter(|leaf| compares_exactly(leaf.conversion()))?;
                        let scale = DataType::arrow_value_scale(arrow_field.dtype());
                        let file_dtype = DataType::from_arrow_field(arrow_field);
                        let lo = Bound::from_scalar(lo, &file_dtype)?;
                        let hi = Bound::from_scalar(hi, &file_dtype)?;
                        Some((leaf, scale, lo, hi))
                    });
                let Some((leaf, scale, lo, hi)) = resolved else {
                    continue;
                };
                for (i, rg) in row_groups.iter().enumerate() {
                    if !is_kept(&mask, i) {
                        continue;
                    }
                    let (min, max) = leaf.bounds(rg, &metadata.footer_buf);
                    let min = min.map(|min| Bound::from_arrow(min, scale));
                    let max = max.map(|max| Bound::from_arrow(max, scale));
                    if matches!(min, Some(None)) || matches!(max, Some(None)) {
                        continue;
                    }
                    if max.flatten().is_some_and(|max| max < lo)
                        || min.flatten().is_some_and(|min| min > hi)
                    {
                        mask.get_or_insert_with(|| {
                            MutableBitmap::from_len_zeroed(row_groups.len())
                        })
                        .set(i, true);
                    }
                }
                false
            },
        };
        if skip_all {
            mask = Some(MutableBitmap::from_len_set(row_groups.len()));
        }
    }

    mask.filter(|m| m.set_bits() > 0).map(MutableBitmap::freeze)
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
    metadata: &FileMetadata,
    row_groups: &[RowGroupMetadata],
    leaf_idxs: &[usize],
    cursor: &mut usize,
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
                let Some(child) =
                    build_struct_statistics_arrays(child, metadata, row_groups, leaf_idxs, cursor)?
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

            match deserialize_all(
                field,
                row_groups,
                idx,
                metadata.column_order(idx),
                &metadata.footer_buf,
            )? {
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
    metadata: &FileMetadata,
    row_groups: &[RowGroupMetadata],
    leaf_idxs: &[usize],
) -> PolarsResult<Option<StatisticsColumns>> {
    let mut cursor = 0;
    let Some(StructStatisticsArrays {
        min,
        max,
        null_count,
    }) = build_struct_statistics_arrays(arrow_field, metadata, row_groups, leaf_idxs, &mut cursor)?
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
    metadata: &FileMetadata,
    row_group_slice: Range<usize>,
    projection: &ArrowFieldProjection,
) -> PolarsResult<StatisticsColumns> {
    let arrow_field = projection.arrow_field();
    let row_groups = &metadata.row_groups[row_group_slice];

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
            load_struct_column_statistics(arrow_field, metadata, row_groups, idxs)?
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

    let Some(statistics) = deserialize_all(
        arrow_field,
        row_groups,
        idx,
        metadata.column_order(idx),
        &metadata.footer_buf,
    )?
    else {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_bounds_compare_as_polars_stores_them() {
        let int = |v: i128| Bound::<&[u8]>::Int(v);
        assert!(Bound::from_arrow(ArrowBound::Int64(7), 1_000).unwrap() == int(7_000));
        assert!(Bound::from_arrow(ArrowBound::Int32(-7), 1_000_000).unwrap() == int(-7_000_000));
        assert!(
            Bound::from_arrow(ArrowBound::UInt64(u64::MAX), 1).unwrap() > int(i64::MAX as i128)
        );
        // A scaled value the reader wraps, and types with no exact comparison.
        assert!(Bound::from_arrow(ArrowBound::Int64(i64::MAX / 1_000 + 1), 1_000).is_none());
        assert!(Bound::from_arrow(ArrowBound::Int64(i64::MIN / 1_000 - 1), 1_000).is_none());
        assert!(Bound::from_arrow(ArrowBound::Float64(1.0), 1).is_none());
        // Bytes and integers have no order between them.
        let bytes = Bound::from_arrow(ArrowBound::Str("abc"), 1).unwrap();
        assert!(bytes < Bound::Bytes(b"abd".to_vec()));
        assert!(bytes.partial_cmp(&int(0)).is_none());
    }
}
