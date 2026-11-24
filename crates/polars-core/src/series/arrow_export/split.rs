use std::ops::Range;

use polars_arrow::array::{Array, FixedSizeListArray, LargeUnionArray, ListArray, StructArray};
use polars_arrow::datatypes::ArrowDataType;

/// The most entries an arrow MAP array, or values of one variant a dense arrow Union, can hold,
/// as their offsets are `i32`.
const MAX_I32_SPAN: u64 = i32::MAX as u64;

/// Whether exporting to `dtype` builds an arrow MAP or dense Union, whose offsets are `i32`.
pub(crate) fn contains_i32_offsets(dtype: &ArrowDataType) -> bool {
    match dtype {
        ArrowDataType::Map(_, _) => true,
        ArrowDataType::Union(u) => {
            u.mode.is_dense() || u.fields.iter().any(|f| contains_i32_offsets(f.dtype()))
        },
        ArrowDataType::LargeList(field) | ArrowDataType::FixedSizeList(field, _) => {
            contains_i32_offsets(field.dtype())
        },
        ArrowDataType::Struct(fields) => fields.iter().any(|f| contains_i32_offsets(f.dtype())),
        ArrowDataType::Extension(ext) => contains_i32_offsets(&ext.inner),
        _ => false,
    }
}

/// The most entries any MAP, or rows any dense Union, of `dtype` spans over `rows` of `array`, a
/// Polars physical array laid out as `ToArrowConverter` expects; reads two offsets per list level.
///
/// A span bounds what export keeps, which can be less: a MAP drops the entries of its null rows
/// (`normalize_map_entries`). An array that `dtype` doesn't describe spans nothing, which leaves
/// the mismatch for conversion to report. A Union's rows bound the values of each of its variants,
/// and its variants are followed only where their whole array could overflow, as that scans the
/// Union's types.
fn max_i32_span(array: &dyn Array, rows: Range<usize>, dtype: &ArrowDataType) -> u64 {
    match dtype {
        ArrowDataType::Map(entries, _) | ArrowDataType::LargeList(entries) => {
            let Some(arr) = array.as_any().downcast_ref::<ListArray<i64>>() else {
                return 0;
            };
            let offsets = arr.offsets().as_slice();
            let child_rows = offsets[rows.start] as usize..offsets[rows.end] as usize;
            let own = match dtype {
                ArrowDataType::Map(_, _) => child_rows.len() as u64,
                _ => 0,
            };
            own.max(max_i32_span(
                arr.values().as_ref(),
                child_rows,
                entries.dtype(),
            ))
        },
        ArrowDataType::FixedSizeList(item, width) => {
            let Some(arr) = array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .filter(|arr| arr.size() == *width)
            else {
                return 0;
            };
            let child_rows = rows.start * width..rows.end * width;
            max_i32_span(arr.values().as_ref(), child_rows, item.dtype())
        },
        ArrowDataType::Struct(fields) => {
            let Some(arr) = array.as_any().downcast_ref::<StructArray>() else {
                return 0;
            };
            arr.values()
                .iter()
                .zip(fields)
                .map(|(values, f)| max_i32_span(values.as_ref(), rows.clone(), f.dtype()))
                .max()
                .unwrap_or(0)
        },
        ArrowDataType::Extension(ext) => max_i32_span(array, rows, &ext.inner),
        ArrowDataType::Union(u) if u.mode.is_dense() => {
            let own = rows.len() as u64;
            let Some(arr) = array.as_any().downcast_ref::<LargeUnionArray>() else {
                return own;
            };
            let types = &arr.types()[rows.clone()];
            let offsets = &arr.offsets()[rows];
            arr.fields()
                .iter()
                .zip(&u.fields)
                .enumerate()
                .filter(|(_, (child, f))| {
                    contains_i32_offsets(f.dtype())
                        && !i32_spans_fit(child.as_ref(), f.dtype(), 0..child.len(), 1)
                })
                .map(|(t, (child, f))| {
                    // A child's offsets increase, so its rows here span it from the first to the
                    // last, the range that export slices or compacts it to.
                    let Some(first) = types.iter().position(|&x| x as usize == t) else {
                        return 0;
                    };
                    let last = types.iter().rposition(|&x| x as usize == t).unwrap();
                    let child_rows = offsets[first] as usize..offsets[last] as usize + 1;
                    max_i32_span(child.as_ref(), child_rows, f.dtype())
                })
                .fold(own, u64::max)
        },
        _ => 0,
    }
}

/// Whether every arrow MAP and dense Union that exporting `rows` of `array` to `dtype` builds fits
/// in `i32` offsets, with `rows` repeated `repeat` times, as a scalar column repeats its one row.
pub(crate) fn i32_spans_fit(
    array: &dyn Array,
    dtype: &ArrowDataType,
    rows: Range<usize>,
    repeat: usize,
) -> bool {
    max_i32_span(array, rows, dtype).saturating_mul(repeat as u64) <= MAX_I32_SPAN
}

/// The end of the longest range from `rows.start` within `rows` that `fits`, or the single row
/// at `rows.start`, which conversion must then check.
///
/// `fits` must stay false once a range doesn't fit, as spans only grow with the range; this
/// calls it `log2(rows.len())` times.
pub(crate) fn arrow_export_cut(rows: Range<usize>, fits: impl Fn(Range<usize>) -> bool) -> usize {
    if rows.is_empty() || fits(rows.clone()) {
        return rows.end;
    }
    let (mut end, mut hi) = (rows.start + 1, rows.end - 1);
    while end < hi {
        let mid = end + (hi - end).div_ceil(2);
        if fits(rows.start..mid) {
            end = mid;
        } else {
            hi = mid - 1;
        }
    }
    end
}

#[cfg(all(test, feature = "dtype-map"))]
mod tests {
    use std::borrow::Cow;

    #[cfg(feature = "dtype-array")]
    use polars_arrow::array::FixedSizeListArray;
    use polars_arrow::array::{BooleanArray, Int32Array, MapArray, NullArray};
    use polars_arrow::bitmap::Bitmap;
    use polars_arrow::offset::Offsets;

    use super::arrow_export_cut;
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

    /// A `Union({"m": Map(Boolean, Null), "i": Int32})` of one chunk whose row `r` holds
    /// variant `types[r]`; the `m` rows take the maps of `lens` in order.
    #[cfg(feature = "dtype-union")]
    fn union_of_maps(lens: &[i64], types: &[i8], validity: Option<Bitmap>) -> Series {
        use polars_arrow::array::LargeUnionArray;

        let fields = vec![
            Field::new("m".into(), map_dtype()),
            Field::new("i".into(), DataType::Int32),
        ];
        let dtype = DataType::Union(UnionFields::try_new(fields).unwrap());
        let mut next = [0i64; 2];
        let offsets: Vec<i64> = types
            .iter()
            .map(|&t| {
                next[t as usize] += 1;
                next[t as usize] - 1
            })
            .collect();
        let ints = Int32Array::from_vec(vec![0; next[1] as usize]).boxed();
        let arr = LargeUnionArray::new(
            dtype.to_physical_arrow(CompatLevel::newest()),
            types.to_vec().into(),
            offsets.into(),
            vec![map_chunk(lens, None), ints],
            validity,
        );
        unsafe { Series::from_chunks_and_dtype_unchecked("u".into(), vec![arr.boxed()], &dtype) }
    }

    /// The rows of each piece, and the most entries of a piece's MAP child.
    #[cfg(feature = "dtype-union")]
    fn union_pieces(s: &Series) -> Vec<(usize, i32)> {
        use polars_arrow::array::UnionArray;

        s.try_to_arrow_split(0, CompatLevel::newest())
            .map(|a| {
                let a = a.unwrap();
                let u: &UnionArray = a.as_any().downcast_ref().unwrap();
                let m: &MapArray = u.fields()[0].as_any().downcast_ref().unwrap();
                (u.len(), *m.offsets().last())
            })
            .collect()
    }

    #[cfg(feature = "dtype-union")]
    #[test]
    fn a_union_is_split_on_the_entries_of_its_maps() {
        let half = i32::MAX as i64 / 2 + 1;
        // The `i` row between the maps must not make their child ranges overlap.
        let s = union_of_maps(&[half, half], &[0, 1, 0], None);

        assert_eq!(union_pieces(&s), [(2, half as i32), (1, half as i32)]);
        let df = s.into_frame();
        let batches: Vec<_> = df
            .try_iter_chunks(CompatLevel::newest(), false)
            .map(|b| b.unwrap().len())
            .collect();
        assert_eq!(batches, [2, 1]);
    }

    /// A union masks the child values of its null rows, and a MAP drops the entries of null rows.
    #[cfg(feature = "dtype-union")]
    #[test]
    fn a_null_union_row_with_too_many_map_entries_exports() {
        let lens = [1, i32::MAX as i64 + 1];
        let s = union_of_maps(&lens, &[0, 0], Some(Bitmap::from([true, false])));

        assert_eq!(union_pieces(&s), [(1, 1), (1, 0)]);
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

    /// A valid row reaches its maps through a null list, whose nulls must be pushed down to them.
    #[test]
    fn a_valid_row_over_a_null_list_of_too_many_map_entries_exports() {
        let half = 1i64 << 30;
        let maps = map_chunk(&[half, half, 1], None);
        let list = ListArray::<i64>::new(
            ListArray::<i64>::default_datatype(maps.dtype().clone()),
            Offsets::try_from_lengths([2, 1].into_iter())
                .unwrap()
                .into(),
            maps,
            Some(Bitmap::from([false, true])),
        );
        let dtype = DataType::Struct(vec![Field::new(
            "l".into(),
            DataType::List(Box::new(map_dtype())),
        )]);
        let rows = StructArray::new(
            dtype.to_physical().to_arrow(CompatLevel::newest()),
            2,
            vec![list.boxed()],
            None,
        );
        let s = unsafe {
            Series::from_chunks_and_dtype_unchecked("s".into(), vec![rows.boxed()], &dtype)
        };

        let arrays: Vec<_> = s
            .try_to_arrow_split(0, CompatLevel::newest())
            .collect::<PolarsResult<_>>()
            .unwrap();

        // The maps under the null list are exported empty.
        let lists: Vec<&ListArray<i64>> = arrays
            .iter()
            .map(|a| {
                let rows: &StructArray = a.as_any().downcast_ref().unwrap();
                rows.values()[0].as_any().downcast_ref().unwrap()
            })
            .collect();
        assert_eq!(lists[0].null_count(), 1);
        let maps: Vec<&MapArray> = lists
            .iter()
            .map(|l| l.values().as_any().downcast_ref().unwrap())
            .collect();
        let offsets: Vec<_> = maps.iter().map(|m| m.offsets().as_slice()).collect();
        assert_eq!(offsets, [&[0, 0, 0][..], &[0, 1]]);

        // A one-row frame holds the row as a Scalar column.
        let df = DataFrame::new_infer_height(vec![Column::from(s.slice(0..1))]).unwrap();
        assert!(df.columns()[0].as_scalar_column().is_some());
        let batches: Vec<_> = df
            .try_iter_chunks(CompatLevel::newest(), false)
            .collect::<PolarsResult<_>>()
            .unwrap();
        assert_eq!(batches.len(), 1);
        let rows: &StructArray = batches[0].arrays()[0].as_any().downcast_ref().unwrap();
        assert_eq!(rows.values()[0].null_count(), 1);
    }

    #[cfg(feature = "dtype-array")]
    #[test]
    fn a_null_array_row_over_too_many_map_entries_exports() {
        let half = 1i64 << 30;
        let array_dtype = DataType::Array(Box::new(map_dtype()), 2);
        let fsl = FixedSizeListArray::new(
            array_dtype.to_physical().to_arrow(CompatLevel::newest()),
            2,
            map_chunk(&[half, half, 1, 1], None),
            Some(Bitmap::from([false, true])),
        );
        let s = unsafe {
            Series::from_chunks_and_dtype_unchecked("a".into(), vec![fsl.boxed()], &array_dtype)
        };

        let arrays: Vec<_> = s
            .try_to_arrow_split(0, CompatLevel::newest())
            .collect::<PolarsResult<_>>()
            .unwrap();

        // The maps under the null row are exported empty.
        let maps: Vec<&MapArray> = arrays
            .iter()
            .map(|a| {
                let fsl: &FixedSizeListArray = a.as_any().downcast_ref().unwrap();
                fsl.values().as_any().downcast_ref().unwrap()
            })
            .collect();
        let offsets: Vec<_> = maps.iter().map(|m| m.offsets().as_slice()).collect();
        assert_eq!(offsets, [[0, 0, 0], [0, 1, 2]]);
        assert_eq!(arrays[0].null_count(), 1);
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

        let mut batches = RecordBatchIter::new(Cow::Borrowed(&df), CompatLevel::newest(), false);

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

    #[test]
    fn a_one_row_chunk_with_too_many_map_entries_errors() {
        let m = map_series(vec![
            map_chunk(&[1], None),
            map_chunk(&[u32::MAX as i64], None),
        ]);
        let df = DataFrame::new_infer_height(vec![m.into_column()]).unwrap();

        let batches: Vec<_> = df.try_iter_chunks(CompatLevel::newest(), false).collect();

        assert_eq!(batches.len(), 2);
        assert!(batches[0].is_ok());
        let err = batches[1].as_ref().unwrap_err();
        assert!(matches!(err, PolarsError::InvalidOperation(_)), "{err}");
    }

    #[test]
    fn a_target_dtype_that_does_not_match_the_array_errors() {
        let s = Series::new("l".into(), [Series::new("".into(), [1i64])]);
        let int = ArrowField::new("item".into(), ArrowDataType::Int64, true);
        let list = ArrowField::new("item".into(), ArrowDataType::LargeList(Box::new(int)), true);
        let map = ArrowField::new(
            "item".into(),
            map_dtype().to_arrow(CompatLevel::newest()),
            true,
        );

        for item in [list, map] {
            let field = ArrowField::new("l".into(), ArrowDataType::LargeList(Box::new(item)), true);
            assert!(s.to_arrow_with_field(0, Cow::Owned(field), true).is_err());
        }
    }

    #[cfg(feature = "dtype-array")]
    #[test]
    fn a_target_width_that_does_not_match_the_array_errors() {
        let map = map_chunk(&[1], None);
        let array_dtype = DataType::Array(Box::new(map_dtype()), 1);
        let fsl = FixedSizeListArray::new(
            array_dtype.to_physical().to_arrow(CompatLevel::newest()),
            1,
            map,
            None,
        );
        let list = ListArray::<i64>::new(
            ListArray::<i64>::default_datatype(fsl.dtype().clone()),
            Offsets::try_from_lengths([1].into_iter()).unwrap().into(),
            fsl.boxed(),
            None,
        );
        let s = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                "l".into(),
                vec![list.boxed()],
                &DataType::List(Box::new(array_dtype)),
            )
        };
        let map = ArrowField::new(
            "item".into(),
            map_dtype().to_arrow(CompatLevel::newest()),
            true,
        );
        let item = ArrowField::new(
            "item".into(),
            ArrowDataType::FixedSizeList(Box::new(map), 2),
            true,
        );
        let field = ArrowField::new("l".into(), ArrowDataType::LargeList(Box::new(item)), true);

        assert!(s.to_arrow_with_field(0, Cow::Owned(field), true).is_err());
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
        let batches = RecordBatchIter::new(Cow::Borrowed(&df), CompatLevel::newest(), false);
        let mut ends = vec![];
        let mut start = 0;
        while start < df.height() {
            start = arrow_export_cut(start..df.height(), |r| batches.fits(&df, 0, r));
            ends.push(start);
        }

        // 2047 rows of 2^20 entries is the most that fits in `i32` offsets.
        assert_eq!(ends, [2047, 4094, 4096]);
        let sc = df.columns()[1].as_scalar_column().unwrap();
        assert!(sc.lazy_as_materialized_series().is_none());
    }
}

#[cfg(all(test, feature = "dtype-union"))]
mod union_tests {
    use std::borrow::Cow;

    use crate::frame::RecordBatchIter;
    use crate::frame::column::ScalarColumn;
    use crate::prelude::*;

    /// A Scalar column repeats its one row without allocating, which stands in for a union chunk
    /// with more rows than the `i32` offsets of an arrow Union can index.
    #[test]
    fn a_union_chunk_is_cut_at_i32_max_rows() {
        let fields = vec![Field::new("i".into(), DataType::Int32)];
        let dtype = DataType::Union(UnionFields::try_new(fields).unwrap());
        let row = UnionChunked::from_variant(&dtype, 0, &Series::new("u".into(), [1i32]))
            .unwrap()
            .into_series();
        let height = i32::MAX as usize + 1;
        let u = ScalarColumn::from_single_value_series(row, height).into_column();
        let df = DataFrame::new(height, vec![u]).unwrap();

        let batches = RecordBatchIter::new(Cow::Borrowed(&df), CompatLevel::newest(), false);

        assert!(batches.fits(&df, 0, 0..height - 1));
        assert!(!batches.fits(&df, 0, 0..height));
    }
}
