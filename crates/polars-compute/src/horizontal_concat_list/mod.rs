#![allow(unsafe_op_in_unsafe_fn)]
use arrow::array::{Array, ListArray};
use arrow::array::builder::{ArrayBuilder, ShareStrategy, make_builder};
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::offset::{Offsets, OffsetsBuffer};
use polars_error::PolarsResult;

/// A contiguous run of inner values taken from a single input column: values `[start, end)` of column `col`.
#[derive(Clone, Copy)]
struct Span {
    col: usize,
    start: usize,
    end: usize,
}

impl Span {
    fn len(&self) -> usize {
        self.end - self.start
    }
}

/// Describes which runs of inner values make up the concatenated output, in row-major output order. Spans are generated
/// lazily by [`SpanPlan::iter`] rather than materialized, so replaying the plan costs no `O(rows * columns)` allocation.
struct SpanPlan<'a> {
    offsets: Vec<&'a OffsetsBuffer<i64>>,
    validities: Vec<Option<&'a Bitmap>>,
    is_broadcast: Vec<bool>,
    output_height: usize,
}

impl SpanPlan<'_> {
    /// Whether output row `row` is valid, i.e. whether every input list at that row is valid (null propagation).
    ///
    /// # Safety
    /// `row < self.output_height` and the preconditions of [`horizontal_concat_list_unchecked`] hold.
    unsafe fn row_is_valid(&self, row: usize) -> bool {
        self.validities
            .iter()
            .zip(&self.is_broadcast)
            .all(|(validity, &broadcast)| {
                let row = if broadcast { 0 } else { row };
                validity.is_none_or(|v| v.get_bit_unchecked(row))
            })
    }

    /// Lazily yields the non-empty spans of every valid output row, in row-major output order.
    ///
    /// # Safety
    /// The preconditions of [`horizontal_concat_list_unchecked`] must hold while the returned iterator is consumed.
    unsafe fn iter(&self) -> impl Iterator<Item = Span> + '_ {
        (0..self.output_height)
            .filter(|&row| self.row_is_valid(row))
            .flat_map(move |row| {
                self.offsets
                    .iter()
                    .zip(&self.is_broadcast)
                    .enumerate()
                    .filter_map(move |(col, (offsets, &broadcast))| {
                        let row = if broadcast { 0 } else { row };
                        let (start, end) = offsets.start_end(row);
                        (end > start).then_some(Span { col, start, end })
                    })
            })
    }
}

/// Horizontally concatenate variable-length list columns row-wise.
///
/// For every output row `r`, the resulting list is the concatenation of
/// `arrays[0][r] ++ arrays[1][r] ++ .. ++ arrays[n-1][r]`. If any input list at row `r` is null, the whole output
/// row is null (null propagation).
///
/// The concatenated inner values are assembled with a single generic [`make_builder`] that bulk-copies each span
/// (a `memcpy` for primitives, buffer sharing for views, per-field recursion for structs), so every inner type goes
/// through one code path.
///
/// # Safety
/// * `arrays` is non-empty.
/// * All arrays in `arrays` have the same inner values type.
/// * All arrays with non-unit length have the same length. Unit-length arrays are broadcast to that common length.
///
/// # Errors
/// Errors if the total number of inner values overflows `i64`.
pub unsafe fn horizontal_concat_list_unchecked(
    arrays: &[ListArray<i64>],
) -> PolarsResult<ListArray<i64>> {
    assert!(!arrays.is_empty());
    debug_assert!(
        arrays
            .iter()
            .all(|a| a.values().dtype() == arrays[0].values().dtype())
    );

    // The output height is the common (broadcast) length: the length of the non-unit columns, or 1 when every column
    // has unit length. Using the non-unit max (rather than the plain max) keeps empty inputs empty even
    // when unit-length columns are broadcast alongside them.
    let output_height = arrays
        .iter()
        .map(|a| a.len())
        .filter(|&l| l != 1)
        .max()
        .unwrap_or(1);
    debug_assert!(
        arrays
            .iter()
            .all(|a| a.len() == output_height || a.len() == 1)
    );

    let plan = SpanPlan {
        offsets: arrays.iter().map(|a| a.offsets()).collect(),
        validities: arrays.iter().map(|a| a.validity()).collect(),
        // A unit-length column is broadcast (unless the output itself has height 1).
        is_broadcast: arrays
            .iter()
            .map(|a| a.len() == 1 && output_height != 1)
            .collect(),
        output_height,
    };

    let has_validity = plan.validities.iter().any(|v| v.is_some());
    let mut out_validity = has_validity.then(|| BitmapBuilder::with_capacity(output_height));

    let mut row_lengths: Vec<usize> = Vec::with_capacity(output_height);
    let mut out_values_len = 0usize;
    for row in 0..output_height {
        let valid = plan.row_is_valid(row);
        if let Some(b) = out_validity.as_mut() {
            b.push(valid);
        }
        if !valid {
            row_lengths.push(0);
            continue;
        }

        let row_len: usize = plan
            .offsets
            .iter()
            .zip(&plan.is_broadcast)
            .map(|(offsets, &broadcast)| {
                let row = if broadcast { 0 } else { row };
                let (start, end) = offsets.start_end(row);
                end - start
            })
            .sum();
        row_lengths.push(row_len);
        out_values_len += row_len;
    }

    // Assemble the concatenated inner values by bulk-copying each span. `make_builder` dispatches on the inner
    // physical type once and handles primitives, views, nested lists, and structs (recursively) uniformly.
    let mut builder = make_builder(arrays[0].values().dtype());
    builder.reserve(out_values_len);
    for span in plan.iter() {
        builder.subslice_extend(
            arrays[span.col].values().as_ref(),
            span.start,
            span.len(),
            ShareStrategy::Always,
        );
    }
    let out_inner = builder.freeze();

    let out_offsets: OffsetsBuffer<i64> =
        Offsets::try_from_lengths(row_lengths.into_iter())?.into();

    Ok(ListArray::<i64>::new(
        arrays[0].dtype().clone(),
        out_offsets,
        out_inner,
        out_validity.and_then(|b| b.into_opt_validity()),
    ))
}
