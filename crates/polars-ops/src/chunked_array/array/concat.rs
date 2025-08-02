use arrow::array::builder::{ShareStrategy, make_builder};
use arrow::array::{Array, FixedSizeListArray};
use arrow::bitmap::BitmapBuilder;
use polars_core::prelude::arity::binary_elementwise;
use polars_core::prelude::*;

pub fn array_concat(left: &ArrayChunked, right: &ArrayChunked) -> PolarsResult<ArrayChunked> {
    // Early validation
    polars_ensure!(
        left.len() == right.len(),
        length_mismatch = "arr.concat", left.len(), right.len()
    );
    
    polars_ensure!(
        left.inner_dtype() == right.inner_dtype(),
        ComputeError: "cannot concatenate arrays with different inner types: {} and {}", 
        left.inner_dtype(), right.inner_dtype()
    );

    
    todo!();

    Ok(result.with_dtype(new_dtype))
}

fn concat_fixed_size_list_arrays(
    left_arr: &FixedSizeListArray,
    right_arr: &FixedSizeListArray,
    left_width: usize,
    right_width: usize,
    new_width: usize,
) -> FixedSizeListArray {
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

        // Append values from the left array.
        let left_start = row * left_width;
        builder.subslice_extend(&**left_values, left_start, left_width, ShareStrategy::Always);

        // Append values from the right array.
        let right_start = row * right_width;
        builder.subslice_extend(&**right_values, right_start, right_width, ShareStrategy::Always);
    }

    let values = builder.freeze();

    // Create a new Arrow data type with the correct concatenated width.
    let new_dtype = ArrowDataType::FixedSizeList(
        Box::new(left_arr.value().clone()),
        new_width,
    );
    
    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars_core::prelude::*;

    #[test]
    fn test_array_concat_basic() -> PolarsResult<()> {
        let left = Series::new("left", &[
            Some(vec![1, 2]),
            Some(vec![3, 4]),
        ]).cast(&DataType::Array(Box::new(DataType::Int32), 2))?;
        
        let right = Series::new("right", &[
            Some(vec![5, 6, 7]),
            Some(vec![8, 9, 10]),
        ]).cast(&DataType::Array(Box::new(DataType::Int32), 3))?;
        
        let result = left.array()?.array_concat(&right)?;
        
        // Should have width 5 (2 + 3)
        assert_eq!(result.array()?.width(), 5);
        assert_eq!(result.len(), 2);
        
        Ok(())
    }

    #[test]
    fn test_array_concat_with_nulls() -> PolarsResult<()> {
        let left = Series::new("left", &[
            Some(vec![1, 2]),
            None::<Vec<i32>>,
        ]).cast(&DataType::Array(Box::new(DataType::Int32), 2))?;
        
        let right = Series::new("right", &[
            Some(vec![5, 6]),
            Some(vec![7, 8]),
        ]).cast(&DataType::Array(Box::new(DataType::Int32), 2))?;
        
        let result = left.array()?.array_concat(&right)?;
        
        // First row should be [1, 2, 5, 6], second should be null
        assert!(!result.is_null(0));
        assert!(result.is_null(1));
        
        Ok(())
    }
}