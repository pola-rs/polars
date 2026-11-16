use std::ops::Range;

use polars_arrow::array::{Array, FixedSizeListArray, ListArray, StructArray};
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

// The walk below follows a Polars physical array along its arrow export dtype, so it relies on
// the layout `ToArrowConverter` relies on: MAP and LARGE_LIST are `ListArray<i64>`, and
// FIXED_SIZE_LIST, STRUCT and EXTENSION wrap the arrays their arrow types describe.

/// Push, for every MAP exported from `rows` of `array` to `dtype` whose offsets there span more
/// than `i32::MAX` entries, the running count of the entries it exports over those rows
/// (`rows.len() + 1` values from 0).
fn push_entry_counts(
    array: &dyn Array,
    rows: Range<usize>,
    dtype: &ArrowDataType,
    out: &mut Vec<Vec<i64>>,
) {
    if !contains_arrow_map(dtype) {
        return;
    }

    match dtype {
        ArrowDataType::Map(entries, _) | ArrowDataType::LargeList(entries) => {
            let arr: &ListArray<i64> = array.as_any().downcast_ref().unwrap();
            let is_map = matches!(dtype, ArrowDataType::Map(_, _));
            let offsets = &arr.offsets().as_slice()[rows.start..=rows.end];
            let first = offsets[0];
            let child_rows = first as usize..offsets[offsets.len() - 1] as usize;
            // MAP export drops what null rows own (`normalize_map_entries`); LARGE_LIST keeps it.
            let dropped =
                |i: usize| is_map && cfg!(feature = "dtype-map") && !arr.is_valid(rows.start + i);

            if is_map && child_rows.len() as i64 > MAX_MAP_ENTRIES {
                let own = (0..rows.len()).map(|i| {
                    if dropped(i) {
                        0
                    } else {
                        offsets[i + 1] - offsets[i]
                    }
                });
                out.push(running_sum(rows.len(), own));
            }

            let nested_start = out.len();
            push_entry_counts(arr.values().as_ref(), child_rows, entries.dtype(), out);
            for counts in &mut out[nested_start..] {
                let per_row = (0..rows.len()).map(|i| {
                    if dropped(i) {
                        0
                    } else {
                        counts[(offsets[i + 1] - first) as usize]
                            - counts[(offsets[i] - first) as usize]
                    }
                });
                *counts = running_sum(rows.len(), per_row);
            }
        },
        ArrowDataType::FixedSizeList(item, width) => {
            let arr: &FixedSizeListArray = array.as_any().downcast_ref().unwrap();
            let nested_start = out.len();
            let child_rows = rows.start * width..rows.end * width;
            push_entry_counts(arr.values().as_ref(), child_rows, item.dtype(), out);
            for counts in &mut out[nested_start..] {
                *counts = (0..=rows.len()).map(|i| counts[i * width]).collect();
            }
        },
        ArrowDataType::Struct(fields) => {
            let arr: &StructArray = array.as_any().downcast_ref().unwrap();
            for (values, f) in arr.values().iter().zip(fields) {
                push_entry_counts(values.as_ref(), rows.clone(), f.dtype(), out);
            }
        },
        ArrowDataType::Extension(ext) => push_entry_counts(array, rows, &ext.inner, out),
        _ => {},
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

/// Row ranges cutting the `height` rows of `arrays` so that every arrow MAP exported from a range
/// fits in its `i32` offsets; `None` if no cut is needed, and an error if a single row exports
/// more MAP entries than that.
///
/// Each array is a whole chunk paired with the arrow dtype it exports to. Without a MAP that
/// costs a dtype walk, with one two offset reads per list level, unless some MAP's offsets span
/// more than `i32::MAX` entries: only then are its rows counted, into a `Vec<i64>`.
pub(crate) fn arrow_export_row_ranges<'a>(
    arrays: impl IntoIterator<Item = (&'a dyn Array, &'a ArrowDataType)>,
    height: usize,
) -> PolarsResult<Option<Vec<Range<usize>>>> {
    let mut counts = Vec::new();
    for (array, dtype) in arrays {
        push_entry_counts(array, 0..height, dtype, &mut counts);
    }
    if counts.is_empty() {
        return Ok(None);
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

    Ok((ranges.len() > 1).then_some(ranges))
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
