use std::ops::Range;

use polars_arrow::array::{Array, FixedSizeListArray, ListArray, StructArray};
use polars_arrow::datatypes::ArrowDataType;
use polars_error::{PolarsResult, polars_bail};

use crate::frame::column::Column;

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

/// The child elements the rows of `arr` span; `None` if that is the whole child.
fn spanned_child(arr: &ListArray<i64>) -> Option<Box<dyn Array>> {
    let offsets = arr.offsets();
    let first = *offsets.first() as usize;
    let len = offsets.range() as usize;
    (first != 0 || len != arr.values().len()).then(|| arr.values().sliced(first, len))
}

// The functions below walk a Polars physical array along its arrow export dtype, so they rely on
// the layout `ToArrowConverter` relies on: MAP and LARGE_LIST are `ListArray<i64>`, and
// FIXED_SIZE_LIST, STRUCT and EXTENSION wrap the arrays their arrow types describe.

/// Whether some MAP exported from `array` to `dtype` could exceed `i32::MAX` entries.
///
/// Reads only the first and last offset of each list level, an upper bound on what it exports.
fn may_overflow(array: &dyn Array, dtype: &ArrowDataType) -> bool {
    if !contains_arrow_map(dtype) {
        return false;
    }

    match dtype {
        ArrowDataType::Map(entries, _) | ArrowDataType::LargeList(entries) => {
            let arr: &ListArray<i64> = array.as_any().downcast_ref().unwrap();
            if matches!(dtype, ArrowDataType::Map(_, _)) && arr.offsets().range() > MAX_MAP_ENTRIES
            {
                return true;
            }
            let child = spanned_child(arr);
            may_overflow(
                child.as_deref().unwrap_or(arr.values().as_ref()),
                entries.dtype(),
            )
        },
        ArrowDataType::FixedSizeList(item, _) => {
            let arr: &FixedSizeListArray = array.as_any().downcast_ref().unwrap();
            may_overflow(arr.values().as_ref(), item.dtype())
        },
        ArrowDataType::Struct(fields) => {
            let arr: &StructArray = array.as_any().downcast_ref().unwrap();
            arr.values()
                .iter()
                .zip(fields)
                .any(|(values, f)| may_overflow(values.as_ref(), f.dtype()))
        },
        ArrowDataType::Extension(ext) => may_overflow(array, &ext.inner),
        _ => false,
    }
}

fn running_sum(len: usize, row_counts: impl Iterator<Item = i64>) -> Vec<i64> {
    let mut out = Vec::with_capacity(len + 1);
    out.push(0);
    let mut acc = 0;
    out.extend(row_counts.map(|n| {
        acc += n;
        acc
    }));
    out
}

/// Push, for every MAP exported from `array` to `dtype` that may overflow, the running count of
/// the entries it exports over the rows of `array` (`array.len() + 1` values from 0).
fn push_entry_counts(array: &dyn Array, dtype: &ArrowDataType, out: &mut Vec<Vec<i64>>) {
    if !may_overflow(array, dtype) {
        return;
    }

    match dtype {
        ArrowDataType::Map(entries, _) | ArrowDataType::LargeList(entries) => {
            let arr: &ListArray<i64> = array.as_any().downcast_ref().unwrap();
            let is_map = matches!(dtype, ArrowDataType::Map(_, _));
            let offsets = arr.offsets().as_slice();
            let first = offsets[0];
            // MAP export drops what null rows own (`normalize_map_entries`); LARGE_LIST keeps it.
            let dropped = |row: usize| is_map && cfg!(feature = "dtype-map") && !arr.is_valid(row);
            let row_count = |row: usize, counts: &[i64]| {
                if dropped(row) {
                    0
                } else {
                    counts[(offsets[row + 1] - first) as usize]
                        - counts[(offsets[row] - first) as usize]
                }
            };

            if is_map && arr.offsets().range() > MAX_MAP_ENTRIES {
                let own = (0..arr.len()).map(|row| {
                    if dropped(row) {
                        0
                    } else {
                        offsets[row + 1] - offsets[row]
                    }
                });
                out.push(running_sum(arr.len(), own));
            }
            let nested_start = out.len();

            let child = spanned_child(arr);
            push_entry_counts(
                child.as_deref().unwrap_or(arr.values().as_ref()),
                entries.dtype(),
                out,
            );
            for counts in &mut out[nested_start..] {
                *counts = running_sum(arr.len(), (0..arr.len()).map(|row| row_count(row, counts)));
            }
        },
        ArrowDataType::FixedSizeList(item, width) => {
            let arr: &FixedSizeListArray = array.as_any().downcast_ref().unwrap();
            let nested_start = out.len();
            push_entry_counts(arr.values().as_ref(), item.dtype(), out);
            for counts in &mut out[nested_start..] {
                *counts = (0..=arr.len()).map(|row| counts[row * width]).collect();
            }
        },
        ArrowDataType::Struct(fields) => {
            let arr: &StructArray = array.as_any().downcast_ref().unwrap();
            for (values, f) in arr.values().iter().zip(fields) {
                push_entry_counts(values.as_ref(), f.dtype(), out);
            }
        },
        ArrowDataType::Extension(ext) => push_entry_counts(array, &ext.inner, out),
        _ => {},
    }
}

/// Row ranges cutting `height` aligned rows of `columns` so that every arrow MAP exported from a
/// range fits in its `i32` offsets; errors if a single row exports more MAP entries than that.
///
/// Each column must be a single chunk or a scalar, paired with the arrow dtype it exports to.
/// Columns without a MAP cost nothing, and the others two offset reads per list level, unless
/// some MAP's offsets span more than `i32::MAX` entries: only then are its rows counted, into
/// one `Vec<i64>` per such MAP.
pub fn arrow_export_row_ranges<'a>(
    columns: impl IntoIterator<Item = (&'a Column, &'a ArrowDataType)>,
    height: usize,
) -> PolarsResult<Vec<Range<usize>>> {
    let mut counts = Vec::new();
    for (column, dtype) in columns {
        if contains_arrow_map(dtype) {
            let s = column.as_materialized_series();
            assert_eq!(s.n_chunks(), 1, "arrow export splits single chunks");
            push_entry_counts(s.chunks()[0].as_ref(), dtype, &mut counts);
        }
    }

    if counts.is_empty() {
        return Ok(std::iter::once(0..height).collect());
    }

    let mut ranges = Vec::new();
    let mut start = 0;
    for row in 0..height {
        if counts
            .iter()
            .all(|c| c[row + 1] - c[start] <= MAX_MAP_ENTRIES)
        {
            continue;
        }
        if let Some(c) = counts
            .iter()
            .find(|c| c[row + 1] - c[row] > MAX_MAP_ENTRIES)
        {
            polars_bail!(
                InvalidOperation:
                "to_arrow() conversion failed: a single row has {} map entries, more than \
                the i32 offsets of the arrow MAP type can hold",
                c[row + 1] - c[row],
            );
        }
        ranges.push(start..row);
        start = row;
    }
    ranges.push(start..height);

    Ok(ranges)
}

#[cfg(all(test, feature = "dtype-map"))]
mod tests {
    use polars_arrow::array::{BooleanArray, MapArray, NullArray};
    use polars_arrow::bitmap::Bitmap;
    use polars_arrow::offset::Offsets;

    use crate::prelude::*;

    /// Python can't build a MAP row with more than `i32::MAX` entries cheaply: arrow import caps
    /// rows at that, and casting to a Map deduplicates keys.
    #[test]
    fn a_null_row_with_too_many_map_entries_exports() {
        let lens = [1i64, i32::MAX as i64 + 1];
        let n = lens.iter().sum::<i64>() as usize;
        let dtype = DataType::Map(Box::new(DataType::Boolean), Box::new(DataType::Null));
        // Zeroed keys are allocated lazily, so the entries cost no memory until read.
        let entries = StructArray::new(
            dtype
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
        let chunk = ListArray::<i64>::new(
            ListArray::<i64>::default_datatype(entries.dtype().clone()),
            offsets.into(),
            entries.boxed(),
            Some(Bitmap::from([true, false])),
        );
        let s = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                PlSmallStr::from_static("m"),
                vec![chunk.boxed()],
                &dtype,
            )
        };

        let arrays = s.try_to_arrow_split(0, CompatLevel::newest()).unwrap();

        assert_eq!(arrays.len(), 1);
        let map: &MapArray = arrays[0].as_any().downcast_ref().unwrap();
        assert_eq!(map.offsets().as_slice(), [0, 1, 1]);
    }
}
