use std::ops::Range;

use polars_arrow::array::{Array, FixedSizeListArray, ListArray, StructArray};
use polars_arrow::datatypes::ArrowDataType;

/// The most entries an arrow MAP array can hold, as its offsets are `i32`.
const MAX_MAP_ENTRIES: u64 = i32::MAX as u64;

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

/// Whether every arrow MAP that exporting all of `array` to `dtype` builds fits in `i32` offsets.
pub(crate) fn fits_arrow_map(array: &dyn Array, dtype: &ArrowDataType) -> bool {
    max_map_span(array, 0..array.len(), dtype) <= MAX_MAP_ENTRIES
}

/// The most entries any MAP of `dtype` spans over `rows` of `array`, a Polars physical array laid
/// out as `ToArrowConverter` expects; reads two offsets per list level.
///
/// A span bounds what export keeps, which can be less: a MAP drops the entries of its null rows
/// (`normalize_map_entries`).
fn max_map_span(array: &dyn Array, rows: Range<usize>, dtype: &ArrowDataType) -> u64 {
    match dtype {
        ArrowDataType::Map(entries, _) | ArrowDataType::LargeList(entries) => {
            let arr: &ListArray<i64> = array.as_any().downcast_ref().unwrap();
            let offsets = arr.offsets().as_slice();
            let child_rows = offsets[rows.start] as usize..offsets[rows.end] as usize;
            let own = match dtype {
                ArrowDataType::Map(_, _) => child_rows.len() as u64,
                _ => 0,
            };
            own.max(max_map_span(
                arr.values().as_ref(),
                child_rows,
                entries.dtype(),
            ))
        },
        ArrowDataType::FixedSizeList(item, width) => {
            let arr: &FixedSizeListArray = array.as_any().downcast_ref().unwrap();
            let child_rows = rows.start * width..rows.end * width;
            max_map_span(arr.values().as_ref(), child_rows, item.dtype())
        },
        ArrowDataType::Struct(fields) => {
            let arr: &StructArray = array.as_any().downcast_ref().unwrap();
            arr.values()
                .iter()
                .zip(fields)
                .map(|(values, f)| max_map_span(values.as_ref(), rows.clone(), f.dtype()))
                .max()
                .unwrap_or(0)
        },
        ArrowDataType::Extension(ext) => max_map_span(array, rows, &ext.inner),
        _ => 0,
    }
}

/// Row ranges cutting the `height` rows of `columns` for export; `None` if no cut is needed.
///
/// Each range either fits, as every arrow MAP it exports spans at most `i32::MAX` entries, or is
/// a single row, which conversion must check. Ranges are as long as that allows.
///
/// Each column is one array with the arrow dtype it exports to. A `repeated` column holds one row
/// that stands for each of the `height` rows, so a scalar column is planned from one row.
///
/// If `0..height` fits, this costs two offset reads per list level; otherwise that times
/// `log2(height)` per range.
pub(crate) fn arrow_export_row_ranges(
    columns: &[(&dyn Array, &ArrowDataType, bool)],
    height: usize,
) -> Option<Vec<Range<usize>>> {
    if height == 0 {
        return None;
    }
    let fits = |rows: Range<usize>| {
        columns.iter().all(|&(array, dtype, repeated)| {
            let span = if repeated {
                max_map_span(array, 0..1, dtype).saturating_mul(rows.len() as u64)
            } else {
                max_map_span(array, rows.clone(), dtype)
            };
            span <= MAX_MAP_ENTRIES
        })
    };
    if fits(0..height) {
        return None;
    }

    let mut ranges = Vec::new();
    let mut start = 0;
    while start < height {
        // Spans only grow with the range, so binary search for the longest one that fits.
        let (mut end, mut hi) = (start + 1, height);
        while end < hi {
            let mid = end + (hi - end).div_ceil(2);
            if fits(start..mid) {
                end = mid;
            } else {
                hi = mid - 1;
            }
        }
        ranges.push(start..end);
        start = end;
    }
    Some(ranges)
}

#[cfg(all(test, feature = "dtype-map"))]
mod tests {
    use polars_arrow::array::{BooleanArray, Int32Array, MapArray, NullArray};
    use polars_arrow::bitmap::Bitmap;
    use polars_arrow::offset::Offsets;

    use crate::frame::RecordBatchIter;
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

        let arrays: Vec<_> = s
            .try_to_arrow_split(0, CompatLevel::newest())
            .collect::<PolarsResult<_>>()
            .unwrap();

        // The null row's span alone overflows, so it is cut off and normalized on its own.
        let maps: Vec<&MapArray> = arrays
            .iter()
            .map(|a| a.as_any().downcast_ref().unwrap())
            .collect();
        let offsets: Vec<_> = maps.iter().map(|m| m.offsets().as_slice()).collect();
        assert_eq!(offsets, [[0, 1], [0, 0]]);
        assert_eq!(maps[1].null_count(), 1);
    }

    /// Python only hands `try_iter_chunks` frames it has aligned.
    #[test]
    fn a_split_chunk_with_misaligned_columns_errors() {
        let half = i32::MAX as i64 / 2 + 1;
        let m = map_series(vec![
            map_chunk(&[half, half], None),
            map_chunk(&[1, 1], None),
        ]);
        let i = Int32Chunked::from_chunk_iter(
            "i".into(),
            [
                Int32Array::from_slice([0, 1, 2]),
                Int32Array::from_slice([3]),
            ],
        );
        let df = DataFrame::new_infer_height(vec![m.into_column(), i.into_column()]).unwrap();

        let mut batches = RecordBatchIter::new(&df, CompatLevel::newest(), false);

        let err = batches.next().unwrap().unwrap_err();
        assert!(matches!(err, PolarsError::ShapeMismatch(_)), "{err}");
    }

    /// Exporting a one-row piece must not read its row, which on `rt32` panics for a map with
    /// `u32::MAX` entries.
    #[test]
    fn a_one_row_piece_with_too_many_map_entries_errors() {
        let m = map_series(vec![map_chunk(&[1, u32::MAX as i64], None)]);
        let df = DataFrame::new_infer_height(vec![m.into_column()]).unwrap();

        let batches: Vec<_> = df.try_iter_chunks(CompatLevel::newest(), false).collect();

        assert_eq!(batches.len(), 2);
        assert!(batches[0].is_ok());
        let err = batches[1].as_ref().unwrap_err();
        assert!(matches!(err, PolarsError::InvalidOperation(_)), "{err}");
    }

    /// A scalar column repeating one map with `keys`, held in two chunks, over `height` rows.
    fn scalar_map_column(keys: Vec<i32>, height: usize) -> Column {
        let dtype = DataType::Map(Box::new(DataType::Int32), Box::new(DataType::Null));
        let entries_dtype = dtype.map_entries_dtype().unwrap();
        let entries = |keys: &[i32]| {
            let n = keys.len();
            StructArray::new(
                entries_dtype.to_physical().to_arrow(CompatLevel::newest()),
                n,
                vec![
                    Int32Array::from_slice(keys).boxed(),
                    NullArray::new(ArrowDataType::Null, n).boxed(),
                ],
                None,
            )
            .boxed()
        };
        let (a, b) = keys.split_at(keys.len() / 2);
        let entries = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                PlSmallStr::from_static("m"),
                vec![entries(a), entries(b)],
                &entries_dtype,
            )
        };
        assert_eq!(entries.n_chunks(), 2);
        let value = Scalar::new(dtype, AnyValue::Map(entries));
        Column::Scalar(ScalarColumn::new("m".into(), value, height))
    }

    #[test]
    fn a_scalar_map_column_is_split_without_materializing_it() {
        let scalar = scalar_map_column((0..1 << 20).collect(), 4096);
        let small = map_series(vec![map_chunk(&[1; 4096], None)]).with_name("small".into());
        let df = DataFrame::new_infer_height(vec![small.into_column(), scalar]).unwrap();
        let dtypes: Vec<_> = df
            .columns()
            .iter()
            .map(|c| c.dtype().to_arrow(CompatLevel::newest()))
            .collect();
        let dtypes: Vec<_> = dtypes.iter().collect();

        let scalar_rows = df.arrow_export_scalar_rows(&dtypes);
        let ranges = df.arrow_export_ranges(&dtypes, &scalar_rows, 0, 0..4096);

        // 2047 rows of 2^20 entries is the most that fits in `i32` offsets.
        assert_eq!(ranges, Some(vec![0..2047, 2047..4094, 4094..4096]));
        let sc = df.columns()[1].as_scalar_column().unwrap();
        assert!(sc.lazy_as_materialized_series().is_none());
    }
}
