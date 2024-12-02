use arrow::array::FixedSizeListArray;
use arrow::compute::utils::combine_validities_and;
use polars_compute::horizontal_flatten::horizontal_flatten_unchecked;
use polars_core::prelude::{ArrayChunked, Column, CompatLevel, DataType, IntoColumn};
use polars_core::series::Series;
use polars_error::{polars_bail, PolarsResult};
use polars_utils::pl_str::PlSmallStr;

/// Note: The caller must ensure all columns in `args` have the same type.
///
/// # Panics
/// Panics if
/// * `args` is empty
/// * `dtype` is not a `DataType::Array`
pub fn concat_arr(args: &[Column], dtype: &DataType) -> PolarsResult<Column> {
    let DataType::Array(inner_dtype, width) = dtype else {
        panic!("{}", dtype);
    };

    let inner_dtype = inner_dtype.as_ref();
    let width = *width;

    let mut output_height = args[0].len();
    let mut calculated_width = 0;
    let mut mismatch_height = (&PlSmallStr::EMPTY, output_height);
    // If there is a `Array` column with a single NULL, the output will be entirely NULL.
    let mut return_all_null = false;

    let (arrays, widths): (Vec<_>, Vec<_>) = args
        .iter()
        .map(|c| {
            // Handle broadcasting
            if output_height == 1 {
                output_height = c.len();
                mismatch_height.1 = c.len();
            }

            if c.len() != output_height && c.len() != 1 && mismatch_height.1 == output_height {
                mismatch_height = (c.name(), c.len());
            }

            match c.dtype() {
                DataType::Array(inner, width) => {
                    debug_assert_eq!(inner.as_ref(), inner_dtype);

                    let arr = c.array().unwrap().rechunk();

                    return_all_null |=
                        arr.len() == 1 && arr.rechunk_validity().is_some_and(|x| !x.get_bit(0));

                    (arr.rechunk().downcast_into_array().values().clone(), *width)
                },
                dtype => {
                    debug_assert_eq!(dtype, inner_dtype);
                    (
                        c.as_materialized_series().rechunk().into_chunks()[0].clone(),
                        1,
                    )
                },
            }
        })
        .filter(|x| x.1 > 0)
        .inspect(|x| calculated_width += x.1)
        .unzip();

    assert_eq!(calculated_width, width);

    if mismatch_height.1 != output_height {
        polars_bail!(
            ShapeMismatch:
            "concat_arr: length of column '{}' (len={}) did not match length of \
            first column '{}' (len={})",
            mismatch_height.0, mismatch_height.1, args[0].name(), output_height,
        )
    }

    if return_all_null {
        let arr =
            FixedSizeListArray::new_null(dtype.to_arrow(CompatLevel::newest()), output_height);
        return Ok(ArrayChunked::with_chunk(args[0].name().clone(), arr).into_column());
    }

    let outer_validity = args
        .iter()
        // Note: We ignore the validity of non-array input columns, their outer is always valid after
        // being reshaped to (-1, 1).
        .filter(|x| {
            // Unit length validities at this point always contain a single valid, as we would have
            // returned earlier otherwise with `return_all_null`, so we filter them out.
            debug_assert!(x.len() == output_height || x.len() == 1);

            x.dtype().is_array() && x.len() == output_height
        })
        .map(|x| x.as_materialized_series().rechunk_validity())
        .fold(None, |a, b| combine_validities_and(a.as_ref(), b.as_ref()));

    let inner_arr = if output_height == 0 || width == 0 {
        Series::new_empty(PlSmallStr::EMPTY, inner_dtype)
            .into_chunks()
            .into_iter()
            .next()
            .unwrap()
    } else {
        unsafe { horizontal_flatten_unchecked(&arrays, &widths, output_height) }
    };

    let arr = FixedSizeListArray::new(
        dtype.to_arrow(CompatLevel::newest()),
        output_height,
        inner_arr,
        outer_validity,
    );

    Ok(ArrayChunked::with_chunk(args[0].name().clone(), arr).into_column())
}
