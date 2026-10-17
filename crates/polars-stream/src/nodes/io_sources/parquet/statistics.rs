use std::cmp::Ordering;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use polars_arrow::array::{MutablePrimitiveArray, PrimitiveArray};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::pushable::Pushable;
use polars_async::executor::{self, TaskMetricAggregator, TaskPriority};
use polars_core::prelude::*;
use polars_io::RowIndex;
use polars_io::predicates::{
    ColumnIoExpr, ReadOrder, RuntimeRange, RuntimeRangeHint, ScanIOPredicate,
};
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

#[allow(clippy::too_many_arguments)]
pub(super) async fn calculate_row_group_pred_pushdown_skip_mask(
    row_group_slice: Range<usize>,
    use_statistics: bool,
    predicate: Option<&ScanIOPredicate>,
    metadata: &Arc<FileMetadata>,
    projected_arrow_fields: Arc<[ArrowFieldProjection]>,
    row_index: Option<RowIndex>,
    verbose: bool,
    task_metrics: Option<&TaskMetricAggregator>,
    runtime_range_skipper: Option<&RuntimeRangeSkipper>,
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
                task_metrics,
            )
            .await?
        },
        None => None,
    };

    let mask = match runtime_range_skipper {
        None => static_mask,
        Some(skipper) => {
            let runtime_mask = skipper.skip_mask(row_group_slice.clone(), static_mask.as_ref());
            match (static_mask, runtime_mask) {
                (Some(s), Some(r)) => Some(&s | &r),
                (s, r) => s.or(r),
            }
        },
    };
    if verbose && (mask.is_some() || runtime_range_skipper.is_some()) {
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
    task_metrics: Option<&TaskMetricAggregator>,
) -> PolarsResult<Option<Bitmap>> {
    let num_row_groups = row_group_slice.len();
    let metadata = metadata.clone();

    // Note: We are spawning here onto the computational async runtime because the caller is being run
    // on a tokio async thread.
    let skip_row_group_mask = executor::spawn(TaskPriority::High, task_metrics, async move {
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
            statistics.null_count = projection.apply_null_count_transform(statistics.null_count)?;

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

/// The runtime ranges of a scan, resolved against the columns of one file. The
/// ranges can change while the file is read, so row groups are checked again
/// right before they are fetched and decoded.
pub(super) struct RuntimeRangeSkipper {
    metadata: Arc<FileMetadata>,
    hints: Vec<ResolvedHint>,
    verbose: bool,
    /// Row groups skipped after the file was opened.
    skipped_while_reading: AtomicUsize,
}

struct ResolvedHint {
    hint: RuntimeRangeHint,
    /// `None` when the column's statistics cannot be compared with a range.
    leaf: Option<ResolvedLeaf>,
}

struct ResolvedLeaf {
    leaf_idx: usize,
    /// The dtype of the bounds.
    dtype: DataType,
    bounds: ResolvedBounds,
}

enum ResolvedBounds {
    /// The row group statistics of the column.
    Statistics { bounds: LeafBounds, scale: i64 },
    /// The bounds of an order-keeping expression of the column, per row group.
    OrderKeeping(Vec<Option<(i128, i128)>>),
}

impl ResolvedLeaf {
    /// The bounds of a row group, `None` when its values are not known to lie between them.
    #[allow(clippy::type_complexity)]
    fn row_group_bounds<'a>(
        &'a self,
        idx: usize,
        metadata: &'a FileMetadata,
    ) -> Option<(Option<Bound<&'a [u8]>>, Option<Bound<&'a [u8]>>)> {
        match &self.bounds {
            ResolvedBounds::Statistics { bounds, scale } => {
                let (min, max) = bounds.bounds(&metadata.row_groups[idx], &metadata.footer_buf);
                let min = min.and_then(|min| Bound::from_arrow(min, *scale));
                let max = max.and_then(|max| Bound::from_arrow(max, *scale));
                // Scaling wraps unseen values unless both bounds show it does not.
                if *scale != 1 && (min.is_none() || max.is_none()) {
                    return None;
                }
                Some((min, max))
            },
            ResolvedBounds::OrderKeeping(bounds) => {
                let (lo, hi) = bounds[idx]?;
                Some((Some(Bound::Int(lo)), Some(Bound::Int(hi))))
            },
        }
    }
}

/// The bounds of `expr`, an order-keeping expression of the column of `projection`, per row group.
fn order_keeping_bounds(
    expr: &dyn ColumnIoExpr,
    projection: &ArrowFieldProjection,
    statistics: &LeafBounds,
    scale: i64,
    metadata: &FileMetadata,
) -> Option<Vec<Option<(i128, i128)>>> {
    let file_dtype = DataType::from_arrow_field(projection.arrow_field());
    let mut mins = Vec::with_capacity(metadata.row_groups.len());
    let mut maxs = Vec::with_capacity(metadata.row_groups.len());
    for row_group in metadata.row_groups.iter() {
        let (min, max) = statistics.bounds(row_group, &metadata.footer_buf);
        let int = |b: Option<ArrowBound>| match b.and_then(|b| Bound::from_arrow(b, scale)) {
            Some(Bound::Int(v)) => i64::try_from(v).ok(),
            _ => None,
        };
        let (min, max) = (int(min), int(max));
        let known = min.is_some() && max.is_some();
        mins.push(min.filter(|_| known));
        maxs.push(max.filter(|_| known));
    }
    let apply = |values: Vec<Option<i64>>| -> PolarsResult<Series> {
        let values = Series::new(projection.output_name().clone(), values)
            .strict_cast(&file_dtype.to_physical())?
            .strict_cast(&file_dtype)?;
        let column = projection.apply_transform(values.into_column())?;
        let df = DataFrame::new(column.len(), vec![column])?;
        Ok(expr
            .evaluate_io(&df)?
            .as_materialized_series()
            .to_physical_repr()
            .into_owned())
    };
    let (mins, maxs) = (apply(mins).ok()?, apply(maxs).ok()?);
    Some(
        (0..mins.len())
            .map(|i| {
                let a = mins.get(i).ok()?.extract::<i128>()?;
                let b = maxs.get(i).ok()?.extract::<i128>()?;
                Some((a.min(b), a.max(b)))
            })
            .collect(),
    )
}

/// What one range says about the row groups, read once for many of them.
enum RangeCheck {
    KeepAll,
    SkipAll,
    Bounds {
        leaf: usize,
        lo: Option<Bound<Vec<u8>>>,
        hi: Option<Bound<Vec<u8>>>,
        nulls_match: bool,
    },
}

impl RuntimeRangeSkipper {
    pub(super) fn new(
        hints: &[RuntimeRangeHint],
        metadata: &Arc<FileMetadata>,
        projected_arrow_fields: &[ArrowFieldProjection],
        verbose: bool,
    ) -> Option<Self> {
        if hints.is_empty() {
            return None;
        }
        let resolve = |hint: &RuntimeRangeHint| {
            if hint.constant.is_some() {
                return None;
            }
            let projection = projected_arrow_fields
                .iter()
                .find(|p| p.output_name() == &hint.column)?;
            let arrow_field = projection.arrow_field();
            let idxs = metadata
                .row_groups
                .first()?
                .columns_idxs_under_root_iter(&arrow_field.name)?;
            let [leaf_idx] = idxs else { return None };
            let bounds = LeafBounds::new(arrow_field, metadata, *leaf_idx)
                .filter(|leaf| compares_exactly(leaf.conversion()))?;
            let scale = DataType::arrow_value_scale(arrow_field.dtype());
            let (dtype, bounds) = match &hint.order_keeping_expr {
                None => (
                    DataType::from_arrow_field(arrow_field),
                    ResolvedBounds::Statistics { bounds, scale },
                ),
                Some(expr) => {
                    let bounds =
                        order_keeping_bounds(expr.as_ref(), projection, &bounds, scale, metadata)?;
                    let column_dtype = projection.output_dtype();
                    let dtype = expr
                        .evaluate_io(&DataFrame::empty_with_schema(&Schema::from_iter([(
                            hint.column.clone(),
                            column_dtype.into_owned(),
                        )])))
                        .ok()?
                        .dtype()
                        .clone();
                    (dtype, ResolvedBounds::OrderKeeping(bounds))
                },
            };
            Some(ResolvedLeaf {
                leaf_idx: *leaf_idx,
                dtype,
                bounds,
            })
        };
        let hints = hints
            .iter()
            .map(|hint| ResolvedHint {
                hint: hint.clone(),
                leaf: resolve(hint),
            })
            .collect();
        Some(Self {
            metadata: metadata.clone(),
            hints,
            verbose,
            skipped_while_reading: AtomicUsize::new(0),
        })
    }

    /// Reads every range once.
    fn checks(&self) -> Vec<RangeCheck> {
        self.hints
            .iter()
            .enumerate()
            .map(|(i, resolved)| {
                let range = resolved.hint.source.runtime_range();
                if let Some(value) = &resolved.hint.constant {
                    return match RuntimeRangeHint::constant_matches(&range, value) {
                        Some(false) => RangeCheck::SkipAll,
                        _ => RangeCheck::KeepAll,
                    };
                }
                match range {
                    RuntimeRange::Pending | RuntimeRange::Disabled => RangeCheck::KeepAll,
                    RuntimeRange::Empty => RangeCheck::SkipAll,
                    RuntimeRange::Range {
                        lo,
                        hi,
                        nulls_match,
                    } => {
                        let Some(leaf) = &resolved.leaf else {
                            return RangeCheck::KeepAll;
                        };
                        let bound = |b: Option<Scalar>| match b {
                            None => Some(None),
                            Some(b) => Bound::from_scalar(&b, &leaf.dtype).map(Some),
                        };
                        match (bound(lo), bound(hi)) {
                            (Some(lo), Some(hi)) => RangeCheck::Bounds {
                                leaf: i,
                                lo,
                                hi,
                                nulls_match,
                            },
                            _ => RangeCheck::KeepAll,
                        }
                    },
                }
            })
            .collect()
    }

    fn skips_with(&self, checks: &[RangeCheck], row_group_idx: usize) -> bool {
        checks.iter().any(|check| match check {
            RangeCheck::KeepAll => false,
            RangeCheck::SkipAll => true,
            RangeCheck::Bounds {
                leaf,
                lo,
                hi,
                nulls_match,
            } => {
                let leaf = self.hints[*leaf].leaf.as_ref().unwrap();
                let Some((min, max)) = leaf.row_group_bounds(row_group_idx, &self.metadata) else {
                    return false;
                };
                let below = lo
                    .as_ref()
                    .is_some_and(|lo| max.as_ref().is_some_and(|max| max < lo));
                let above = hi
                    .as_ref()
                    .is_some_and(|hi| min.as_ref().is_some_and(|min| min > hi));
                (below || above)
                    && (!nulls_match
                        || has_no_nulls(&self.metadata.row_groups[row_group_idx], leaf.leaf_idx))
            },
        })
    }

    /// Which row groups of `row_group_slice` the ranges skip, among those
    /// `static_mask` keeps. `None` when they skip nothing.
    fn skip_mask(
        &self,
        row_group_slice: Range<usize>,
        static_mask: Option<&Bitmap>,
    ) -> Option<Bitmap> {
        let checks = self.checks();
        let mut mask = MutableBitmap::from_len_zeroed(row_group_slice.len());
        for (i, row_group_idx) in row_group_slice.enumerate() {
            if !static_mask.is_some_and(|m| m.get_bit(i)) && self.skips_with(&checks, row_group_idx)
            {
                mask.set(i, true);
            }
        }
        (mask.set_bits() > 0).then(|| mask.freeze())
    }

    /// Sorts `items` by the statistics of the first range that has a read order.
    /// Row groups without the statistic go last.
    pub(super) fn sort_by_read_order<T>(&self, items: &mut [T], idx: impl Fn(&T) -> usize) {
        let Some((order, resolved)) = self
            .hints
            .iter()
            .find_map(|resolved| Some((resolved.hint.source.read_order()?, resolved)))
        else {
            return;
        };
        let Some(leaf) = &resolved.leaf else {
            return;
        };
        let key = |item: &T| {
            let (min, max) = leaf.row_group_bounds(idx(item), &self.metadata)?;
            match order {
                ReadOrder::LowFirst => min,
                ReadOrder::HighFirst => max,
            }
        };
        let keys: Vec<_> = items.iter().map(key).collect();
        let mut positions: Vec<usize> = (0..items.len()).collect();
        positions.sort_by(|&l, &r| match (&keys[l], &keys[r]) {
            (Some(l), Some(r)) => {
                let ord = l.partial_cmp(r).unwrap_or(Ordering::Equal);
                match order {
                    ReadOrder::LowFirst => ord,
                    ReadOrder::HighFirst => ord.reverse(),
                }
            },
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        });
        let mut rank = vec![0; self.metadata.row_groups.len()];
        for (r, &pos) in positions.iter().enumerate() {
            rank[idx(&items[pos])] = r;
        }
        items.sort_unstable_by_key(|item| rank[idx(item)]);
        if self.verbose {
            eprintln!(
                "[ParquetFileReader]: Reading row groups by the {} of '{}'",
                match order {
                    ReadOrder::LowFirst => "min",
                    ReadOrder::HighFirst => "max",
                },
                resolved.hint.column
            );
        }
    }

    /// Whether the ranges skip row group `row_group_idx` now.
    pub(super) fn skips(&self, row_group_idx: usize) -> bool {
        let skip = self.skips_with(&self.checks(), row_group_idx);
        if skip {
            self.skipped_while_reading
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
        skip
    }
}

fn has_no_nulls(row_group: &RowGroupMetadata, leaf_idx: usize) -> bool {
    let column = &row_group.parquet_columns()[leaf_idx];
    column.descriptor().descriptor.max_def_level == 0 || column.null_count() == Some(0)
}

impl Drop for RuntimeRangeSkipper {
    fn drop(&mut self) {
        let skipped = *self.skipped_while_reading.get_mut();
        if self.verbose && skipped > 0 {
            eprintln!(
                "[ParquetFileReader]: Runtime ranges skipped {skipped} more row groups while reading"
            );
        }
    }
}

/// Recursively assemble per-field `min` / `max` / `null_count` statistics columns for a
/// (possibly nested) struct field, consuming one parquet leaf column per scalar leaf. Returns
/// `None` (signalling the caller to fall back to null statistics) for an empty struct, an
/// unsupported leaf type (e.g. a nested list), or if the leaves run out before the fields do.
fn build_struct_statistics_columns(
    field: &ArrowField,
    metadata: &FileMetadata,
    row_groups: &[RowGroupMetadata],
    leaf_idxs: &[usize],
    cursor: &mut usize,
) -> PolarsResult<Option<StatisticsColumns>> {
    let height = row_groups.len();
    match field.dtype() {
        ArrowDataType::Struct(children) => {
            // An empty struct has no leaf statistics to reason about.
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
                let Some(child_statistics) = build_struct_statistics_columns(
                    child, metadata, row_groups, leaf_idxs, cursor,
                )?
                else {
                    return Ok(None);
                };
                mins.push(child_statistics.min.with_name(child.name.clone()));
                maxs.push(child_statistics.max.with_name(child.name.clone()));
                ncs.push(child_statistics.null_count.with_name(child.name.clone()));
            }

            // The null-count struct mirrors the field's shape with each leaf replaced by the
            // index type.
            let to_struct = |fields: &[Column]| {
                PolarsResult::Ok(
                    StructChunked::from_columns(PlSmallStr::EMPTY, height, fields)?
                        .into_series()
                        .into_column(),
                )
            };

            Ok(Some(StatisticsColumns {
                min: to_struct(&mins)?,
                max: to_struct(&maxs)?,
                null_count: to_struct(&ncs)?,
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
                // Converted to the logical type here, the statistics arrays are physical.
                Some(statistics) => Ok(Some(StatisticsColumns::from_arrow_statistics(
                    statistics, field,
                )?)),
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
    let Some(statistics) =
        build_struct_statistics_columns(arrow_field, metadata, row_groups, leaf_idxs, &mut cursor)?
    else {
        return Ok(None);
    };

    // Only trust the assembled stats if every parquet leaf mapped to a struct field; a
    // mismatch means the schema and parquet layout disagree and the stats could be misaligned.
    if cursor != leaf_idxs.len() {
        return Ok(None);
    }

    Ok(Some(statistics))
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
        assert!(bytes < Bound::Bytes(b"abz".to_vec()));
        assert!(bytes.partial_cmp(&int(0)).is_none());
    }
}
