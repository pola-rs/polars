#![allow(unsafe_op_in_unsafe_fn)]
use arrow::array::builder::{ArrayBuilder, ShareStrategy, make_builder};
use arrow::array::{Array, ListArray};
use arrow::bitmap::BitmapBuilder;
use arrow::offset::{Offsets, OffsetsBuffer};
use polars_error::PolarsResult;

/// Horizontally concatenate variable-length list columns row-wise.
///
/// For every output row `r`, the resulting list is the concatenation of
/// `arrays[0][r] ++ arrays[1][r] ++ .. ++ arrays[n-1][r]`.
/// If any input list at row `r` is null, the whole output row is null (null propagation).
///
/// The concatenated inner values are assembled with a single generic [`make_builder`] that bulk-copies each non-empty
/// input list (a `memcpy` for primitives, buffer sharing for views, per-field recursion for structs), so every inner
/// type goes through one code path.
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
    // has unit length. Using the non-unit max (rather than the plain max) keeps empty inputs empty even when unit
    // length columns are broadcast alongside them.
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

    let has_validity = arrays.iter().any(|a| a.validity().is_some());
    let mut out_validity = has_validity.then(|| BitmapBuilder::with_capacity(output_height));

    let mut row_lengths: Vec<usize> = Vec::with_capacity(output_height);
    let mut out_values_len = 0usize;
    for row in 0..output_height {
        let valid = arrays.iter().all(|array| {
            let input_row = if array.len() == 1 { 0 } else { row };
            array
                .validity()
                .is_none_or(|v| v.get_bit_unchecked(input_row))
        });
        if let Some(b) = out_validity.as_mut() {
            b.push(valid);
        }
        if !valid {
            row_lengths.push(0);
            continue;
        }

        let row_len: usize = arrays
            .iter()
            .map(|array| {
                let input_row = if array.len() == 1 { 0 } else { row };
                let (start, end) = array.offsets().start_end(input_row);
                end - start
            })
            .sum();
        row_lengths.push(row_len);
        out_values_len += row_len;
    }

    let out_validity = out_validity.and_then(|b| b.into_opt_validity());

    // Assemble the concatenated inner values. `make_builder` dispatches on the inner physical type once and handles
    // primitives, views, nested lists, and structs (recursively) uniformly.
    let mut builder = make_builder(arrays[0].values().dtype());
    builder.reserve(out_values_len);
    for row in 0..output_height {
        if out_validity
            .as_ref()
            .is_some_and(|v| !v.get_bit_unchecked(row))
        {
            continue;
        }

        for array in arrays {
            let input_row = if array.len() == 1 { 0 } else { row };
            let (start, end) = array.offsets().start_end(input_row);
            if start < end {
                builder.subslice_extend(
                    array.values().as_ref(),
                    start,
                    end - start,
                    ShareStrategy::Always,
                );
            }
        }
    }

    let out_inner = builder.freeze();
    let out_offsets: OffsetsBuffer<i64> =
        Offsets::try_from_lengths(row_lengths.into_iter())?.into();

    Ok(ListArray::<i64>::new(
        arrays[0].dtype().clone(),
        out_offsets,
        out_inner,
        out_validity,
    ))
}
