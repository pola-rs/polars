use std::ops::Range;

use polars_arrow::array::{Array, ArrayRef, FixedSizeListArray, ListArray, StructArray};
use polars_arrow::datatypes::ArrowDataType;
use polars_error::{PolarsResult, polars_bail};

/// The most entries an arrow MAP array can hold, as its offsets are `i32`.
const MAX_MAP_ENTRIES: i64 = i32::MAX as i64;

/// Whether exporting to `dtype` builds an arrow MAP, whose offsets are `i32`.
pub(crate) fn contains_arrow_map(dtype: &ArrowDataType) -> bool {
    match dtype {
        ArrowDataType::Map(_, _) => true,
        ArrowDataType::LargeList(field) | ArrowDataType::FixedSizeList(field, _) => {
            contains_arrow_map(field.dtype())
        },
        ArrowDataType::Struct(fields) => fields.iter().any(|f| contains_arrow_map(f.dtype())),
        ArrowDataType::Extension(ext) => contains_arrow_map(&ext.inner),
        _ => false,
    }
}

/// The number of MAP nodes in `dtype`, which the walks below number in pre-order.
fn count_arrow_maps(dtype: &ArrowDataType) -> usize {
    match dtype {
        ArrowDataType::Map(entries, _) => 1 + count_arrow_maps(entries.dtype()),
        ArrowDataType::LargeList(field) | ArrowDataType::FixedSizeList(field, _) => {
            count_arrow_maps(field.dtype())
        },
        ArrowDataType::Struct(fields) => fields.iter().map(|f| count_arrow_maps(f.dtype())).sum(),
        ArrowDataType::Extension(ext) => count_arrow_maps(&ext.inner),
        _ => 0,
    }
}

// The walks below follow a Polars physical array along its arrow export dtype, so they rely on
// the layout `ToArrowConverter` relies on: MAP and LARGE_LIST are `ListArray<i64>`, and
// FIXED_SIZE_LIST, STRUCT and EXTENSION wrap the arrays their arrow types describe.

/// Add to `out[node]` the entries that MAP `node` of `dtype` spans over `rows` of `array`, an
/// upper bound on what export writes; reads two offsets per list level.
fn add_entry_bounds(
    array: &dyn Array,
    rows: Range<usize>,
    dtype: &ArrowDataType,
    out: &mut [i64],
    node: &mut usize,
) {
    match dtype {
        ArrowDataType::Map(entries, _) | ArrowDataType::LargeList(entries) => {
            let arr: &ListArray<i64> = array.as_any().downcast_ref().unwrap();
            let offsets = arr.offsets().as_slice();
            let child_rows = offsets[rows.start] as usize..offsets[rows.end] as usize;
            if matches!(dtype, ArrowDataType::Map(_, _)) {
                out[*node] += child_rows.len() as i64;
                *node += 1;
            }
            add_entry_bounds(
                arr.values().as_ref(),
                child_rows,
                entries.dtype(),
                out,
                node,
            );
        },
        ArrowDataType::FixedSizeList(item, width) => {
            let arr: &FixedSizeListArray = array.as_any().downcast_ref().unwrap();
            let child_rows = rows.start * width..rows.end * width;
            add_entry_bounds(arr.values().as_ref(), child_rows, item.dtype(), out, node);
        },
        ArrowDataType::Struct(fields) => {
            let arr: &StructArray = array.as_any().downcast_ref().unwrap();
            for (values, f) in arr.values().iter().zip(fields) {
                add_entry_bounds(values.as_ref(), rows.clone(), f.dtype(), out, node);
            }
        },
        ArrowDataType::Extension(ext) => add_entry_bounds(array, rows, &ext.inner, out, node),
        _ => {},
    }
}

/// Add to `out[node]` the entries export writes for MAP `node` of `dtype` over `rows` of
/// `array`, for the nodes marked in `hot`; subtrees without a hot node are skipped.
fn add_entry_counts(
    array: &dyn Array,
    rows: Range<usize>,
    dtype: &ArrowDataType,
    hot: &[bool],
    out: &mut [i64],
    node: &mut usize,
) {
    let n_maps = count_arrow_maps(dtype);
    if !hot[*node..*node + n_maps].contains(&true) {
        *node += n_maps;
        return;
    }

    match dtype {
        ArrowDataType::Map(entries, _) | ArrowDataType::LargeList(entries) => {
            let arr: &ListArray<i64> = array.as_any().downcast_ref().unwrap();
            let offsets = arr.offsets().as_slice();
            let map_node = matches!(dtype, ArrowDataType::Map(_, _)).then(|| {
                *node += 1;
                *node - 1
            });
            let child_node = *node;
            let mut add_rows = |rows: Range<usize>, node: &mut usize| {
                let child_rows = offsets[rows.start] as usize..offsets[rows.end] as usize;
                if let Some(m) = map_node {
                    out[m] += child_rows.len() as i64;
                }
                *node = child_node;
                add_entry_counts(
                    arr.values().as_ref(),
                    child_rows,
                    entries.dtype(),
                    hot,
                    out,
                    node,
                );
            };

            // MAP export drops what null rows own (`normalize_map_entries`); LARGE_LIST keeps it.
            let drops_nulls = map_node.is_some() && cfg!(feature = "dtype-map");
            match arr
                .validity()
                .filter(|_| drops_nulls && arr.null_count() > 0)
            {
                None => add_rows(rows, node),
                Some(validity) => {
                    let mut run_start = None;
                    for row in rows.clone() {
                        match (validity.get_bit(row), run_start) {
                            (true, None) => run_start = Some(row),
                            (false, Some(start)) => {
                                add_rows(start..row, node);
                                run_start = None;
                            },
                            _ => {},
                        }
                    }
                    if let Some(start) = run_start {
                        add_rows(start..rows.end, node);
                    }
                    *node = child_node + count_arrow_maps(entries.dtype());
                },
            }
        },
        ArrowDataType::FixedSizeList(item, width) => {
            let arr: &FixedSizeListArray = array.as_any().downcast_ref().unwrap();
            let child_rows = rows.start * width..rows.end * width;
            add_entry_counts(
                arr.values().as_ref(),
                child_rows,
                item.dtype(),
                hot,
                out,
                node,
            );
        },
        ArrowDataType::Struct(fields) => {
            let arr: &StructArray = array.as_any().downcast_ref().unwrap();
            for (values, f) in arr.values().iter().zip(fields) {
                add_entry_counts(values.as_ref(), rows.clone(), f.dtype(), hot, out, node);
            }
        },
        ArrowDataType::Extension(ext) => add_entry_counts(array, rows, &ext.inner, hot, out, node),
        _ => {},
    }
}

/// Row ranges cutting the `height` rows of `columns` so that every arrow MAP exported from a
/// range fits in its `i32` offsets; `None` if no cut is needed, and an error if a single row
/// exports more MAP entries than that.
///
/// Each column is given as its chunks with the arrow dtype it exports to; columns may be chunked
/// differently. A `repeated` column is a single row, whose chunks may each hold part of it, standing
/// for each of the `height` rows, so a scalar column is planned without materializing it. The cost is two offset reads per list level and chunk, unless some MAP's offsets
/// span more than `i32::MAX` entries: only then does a pass over the rows count what each row
/// exports of those MAPs, allocating nothing per row.
pub(crate) fn arrow_export_row_ranges<'a>(
    columns: impl IntoIterator<Item = (&'a [ArrayRef], &'a ArrowDataType, bool)>,
    height: usize,
) -> PolarsResult<Option<Vec<Range<usize>>>> {
    let columns: Vec<_> = columns
        .into_iter()
        .filter(|(_, dtype, _)| contains_arrow_map(dtype))
        .collect();
    let n_maps = columns.iter().map(|(_, d, _)| count_arrow_maps(d)).sum();

    let mut bounds = vec![0i64; n_maps];
    let mut node = 0;
    for (chunks, dtype, repeated) in &columns {
        let first = node;
        for chunk in chunks.iter() {
            node = first;
            add_entry_bounds(
                chunk.as_ref(),
                0..chunk.len(),
                dtype,
                &mut bounds,
                &mut node,
            );
        }
        node = first + count_arrow_maps(dtype);
        if *repeated {
            for b in &mut bounds[first..node] {
                *b *= height as i64;
            }
        }
    }
    let hot: Vec<bool> = bounds.iter().map(|&b| b > MAX_MAP_ENTRIES).collect();
    if !hot.contains(&true) {
        return Ok(None);
    }

    // A repeated column exports the same entries in every row, so count them once.
    let mut repeated_counts = vec![0i64; n_maps];
    let mut node = 0;
    for (chunks, dtype, repeated) in &columns {
        let first = node;
        node += count_arrow_maps(dtype);
        if *repeated {
            for chunk in chunks.iter() {
                let mut chunk_node = first;
                add_entry_counts(
                    chunk.as_ref(),
                    0..chunk.len(),
                    dtype,
                    &hot,
                    &mut repeated_counts,
                    &mut chunk_node,
                );
            }
        }
    }

    // Greedy: extend the range while every MAP fits, as late cuts give the fewest pieces.
    let mut cursors: Vec<(usize, usize)> = vec![(0, 0); columns.len()];
    let mut row_counts = vec![0i64; n_maps];
    let mut range_counts = vec![0i64; n_maps];
    let mut ranges = Vec::new();
    let mut start = 0;
    for row in 0..height {
        row_counts.copy_from_slice(&repeated_counts);
        let mut node = 0;
        for ((chunks, dtype, repeated), (chunk_idx, local)) in columns.iter().zip(&mut cursors) {
            if *repeated {
                node += count_arrow_maps(dtype);
                continue;
            }
            while *local == chunks[*chunk_idx].len() {
                *chunk_idx += 1;
                *local = 0;
            }
            let chunk = chunks[*chunk_idx].as_ref();
            add_entry_counts(
                chunk,
                *local..*local + 1,
                dtype,
                &hot,
                &mut row_counts,
                &mut node,
            );
            *local += 1;
        }

        if let Some(&n) = row_counts.iter().find(|&&n| n > MAX_MAP_ENTRIES) {
            polars_bail!(
                InvalidOperation:
                "to_arrow() conversion failed: a single row has {n} map entries, more than \
                the i32 offsets of the arrow MAP type can hold",
            );
        }
        if range_counts
            .iter()
            .zip(&row_counts)
            .any(|(r, n)| r + n > MAX_MAP_ENTRIES)
        {
            ranges.push(start..row);
            start = row;
            range_counts.fill(0);
        }
        for (r, n) in range_counts.iter_mut().zip(&row_counts) {
            *r += n;
        }
    }
    ranges.push(start..height);

    Ok((ranges.len() > 1).then_some(ranges))
}

#[cfg(all(test, feature = "dtype-map"))]
mod tests {
    use polars_arrow::array::{BooleanArray, MapArray, NullArray};
    use polars_arrow::bitmap::Bitmap;
    use polars_arrow::offset::Offsets;

    use crate::frame::column::ScalarColumn;
    use crate::prelude::*;

    fn map_dtype() -> DataType {
        DataType::Map(Box::new(DataType::Boolean), Box::new(DataType::Null))
    }

    /// A `Map(Boolean, Null)` chunk with `lens` entries per row; its zeroed keys are allocated
    /// lazily, so the entries cost no memory until read.
    fn map_chunk(lens: &[i64], validity: Option<Bitmap>) -> ArrayRef {
        let n = lens.iter().sum::<i64>() as usize;
        let entries = StructArray::new(
            map_dtype()
                .map_entries_dtype()
                .unwrap()
                .to_physical()
                .to_arrow(CompatLevel::newest()),
            n,
            vec![
                BooleanArray::new(ArrowDataType::Boolean, Bitmap::new_zeroed(n), None).boxed(),
                NullArray::new(ArrowDataType::Null, n).boxed(),
            ],
            None,
        );
        let offsets = Offsets::try_from_lengths(lens.iter().map(|&l| l as usize)).unwrap();
        ListArray::<i64>::new(
            ListArray::<i64>::default_datatype(entries.dtype().clone()),
            offsets.into(),
            entries.boxed(),
            validity,
        )
        .boxed()
    }

    fn map_series(chunks: Vec<ArrayRef>) -> Series {
        unsafe {
            Series::from_chunks_and_dtype_unchecked(
                PlSmallStr::from_static("m"),
                chunks,
                &map_dtype(),
            )
        }
    }

    /// Python can't build a MAP row with more than `i32::MAX` entries cheaply: arrow import caps
    /// rows at that, and casting to a Map deduplicates keys.
    #[test]
    fn a_null_row_with_too_many_map_entries_exports() {
        let lens = [1i64, i32::MAX as i64 + 1];
        let s = map_series(vec![map_chunk(&lens, Some(Bitmap::from([true, false])))]);

        let arrays = s.try_to_arrow_split(0, CompatLevel::newest()).unwrap();

        assert_eq!(arrays.len(), 1);
        let map: &MapArray = arrays[0].as_any().downcast_ref().unwrap();
        assert_eq!(map.offsets().as_slice(), [0, 1, 1]);
    }

    #[test]
    fn split_for_arrow_export_cuts_across_chunks_without_rechunking() {
        let row = 1i64 << 29;
        let chunk = map_chunk(&[row, row], None);
        let s = map_series(vec![chunk.clone(), chunk.clone(), chunk]);
        let df = DataFrame::new_infer_height(vec![s.into_column()]).unwrap();
        let arrow_dtype = map_dtype().to_arrow(CompatLevel::newest());

        let pieces = df.split_for_arrow_export([&arrow_dtype]).unwrap();

        let heights: Vec<_> = pieces.iter().map(|p| p.height()).collect();
        assert_eq!(heights, [3, 3]);
        let n_chunks: Vec<_> = pieces.iter().map(|p| p.columns()[0].n_chunks()).collect();
        assert_eq!(n_chunks, [2, 2]);
    }

    #[test]
    fn a_scalar_map_column_is_split_without_materializing_it() {
        // One map of 2^29 entries, held in two chunks.
        let halves = map_series(vec![map_chunk(&[1 << 28, 1 << 28], None)]);
        let entries = |i| match halves.get(i).unwrap() {
            AnyValue::Map(entries) => entries,
            _ => unreachable!(),
        };
        let mut value = entries(0);
        value.append(&entries(1)).unwrap();
        assert_eq!(value.n_chunks(), 2);
        let value = Scalar::new(map_dtype(), AnyValue::Map(value));
        let scalar = Column::Scalar(ScalarColumn::new("m".into(), value, 6));
        let small = map_series(vec![map_chunk(&[1; 6], None)]).with_name("small".into());
        let small = small.into_column();
        let df = DataFrame::new_infer_height(vec![small, scalar]).unwrap();
        let arrow_dtype = map_dtype().to_arrow(CompatLevel::newest());
        let dtypes = [&arrow_dtype, &arrow_dtype];

        let ranges = df.arrow_export_ranges(dtypes, None).unwrap();
        let chunk_ranges = df.arrow_export_ranges(dtypes, Some((0, 0..6))).unwrap();

        assert_eq!(ranges, Some(vec![0..3, 3..6]));
        assert_eq!(chunk_ranges, ranges);
        let sc = df.columns()[1].as_scalar_column().unwrap();
        assert!(sc.lazy_as_materialized_series().is_none());
    }
}
