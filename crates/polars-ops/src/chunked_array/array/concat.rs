use arrow::array::builder::{ArrayBuilder, ShareStrategy, make_builder};
use arrow::array::{Array, FixedSizeListArray};
use arrow::bitmap::BitmapBuilder;
use polars_core::prelude::arity::binary_elementwise;
use polars_core::prelude::*;

pub fn array_concat(left: &ArrayChunked, right: &ArrayChunked) -> PolarsResult<ArrayChunked> {
    // Early validation
    polars_ensure!(
        left.len() == right.len(),
        length_mismatch = "arr.concat",
        left.len(),
        right.len()
    );

    polars_ensure!(
        left.inner_dtype() == right.inner_dtype(),
        ComputeError: "cannot concatenate arrays with different inner types: {} and {}",
        left.inner_dtype(), right.inner_dtype()
    );

    let left_width = left.width();
    let right_width = right.width();
    let new_width = left_width + right_width;

    // Create new dtype with combined width
    let new_dtype = DataType::Array(Box::new(left.inner_dtype().clone()), new_width);

    // Process each chunk
    let chunks: PolarsResult<Vec<_>> = left
        .downcast_iter()
        .zip(right.downcast_iter())
        .map(|(left_arr, right_arr)| {
            concat_fixed_size_list_arrays(left_arr, right_arr, left_width, right_width, new_width)
        })
        .collect();

    let result = ArrayChunked::from_chunk_iter(left.name().clone(), chunks?);
    Ok(result)
}

fn concat_fixed_size_list_arrays(
    left_arr: &FixedSizeListArray,
    right_arr: &FixedSizeListArray,
    left_width: usize,
    right_width: usize,
    new_width: usize,
) -> PolarsResult<FixedSizeListArray> {
    let len = left_arr.len();
    let mut builder = make_builder(left_arr.values().dtype());

    builder.reserve(len * new_width);
    let mut validity = BitmapBuilder::with_capacity(len);

    let left_values = left_arr.values();
    let right_values = right_arr.values();

    // Process each row to build the new concatenated array.
    for row in 0..len {
        let is_valid = left_arr.is_valid(row) && right_arr.is_valid(row);
        validity.push(is_valid);

        if !is_valid {
            // If the row is null in either input, the output row is null.
            builder.extend_nulls(new_width);
            continue;
        }

        let left_start = row * left_width;
        builder.subslice_extend(
            &**left_values,
            left_start,
            left_width,
            ShareStrategy::Always,
        );

        let right_start = row * right_width;
        builder.subslice_extend(
            &**right_values,
            right_start,
            right_width,
            ShareStrategy::Always,
        );
    }

    let values = builder.freeze();

    // Create a new Arrow data type with the correct concatenated width.
    let field = left_arr.dtype();

    let inner_field = if let ArrowDataType::FixedSizeList(inner, _) = field {
        inner.as_ref().clone()
    } else {
        return Err(PolarsError::ComputeError("Expected FixedSizeList".into()));
    };

    let new_dtype = ArrowDataType::FixedSizeList(Box::new(inner_field), new_width);

    let validity_bitmap = validity.freeze();

    Ok(FixedSizeListArray::new(
        new_dtype,
        len,
        values,
        Some(validity_bitmap),
    ))
}
