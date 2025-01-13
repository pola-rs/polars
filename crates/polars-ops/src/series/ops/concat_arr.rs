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
    // Indicates whether all `arrays` have unit length (excluding zero-width arrays)
    let mut all_unit_len = true;
    let mut validities = Vec::with_capacity(args.len());

    let (arrays, widths): (Vec<_>, Vec<_>) = args
        .iter()
        .map(|c| {
            let len = c.len();

            // Handle broadcasting
            if output_height == 1 {
                output_height = len;
                mismatch_height.1 = len;
            }

            if len != output_height && len != 1 && mismatch_height.1 == output_height {
                mismatch_height = (c.name(), len);
            }

            // Don't expand scalars to height, this is handled by the `horizontal_flatten` kernel.
            let s = match c {
                Column::Scalar(s) => s.as_single_value_series(),
                v => v.as_materialized_series().clone(),
            };

            match s.dtype() {
                DataType::Array(inner, width) => {
                    debug_assert_eq!(inner.as_ref(), inner_dtype);

                    let arr = s.array().unwrap().rechunk();
                    let validity = arr.rechunk_validity();

                    return_all_null |= len == 1 && validity.as_ref().is_some_and(|x| !x.get_bit(0));

                    // Ignore unit-length validities. If they are non-valid then `return_all_null` will
                    // cause an early return.
                    if let Some(v) = validity.filter(|_| len > 1) {
                        validities.push(v)
                    }

                    (arr.rechunk().downcast_into_array().values().clone(), *width)
                },
                dtype => {
                    debug_assert_eq!(dtype, inner_dtype);
                    // Note: We ignore the validity of non-array input columns, their outer is always valid after
                    // being reshaped to (-1, 1).
                    (s.rechunk().into_chunks()[0].clone(), 1)
                },
            }
        })
        // Filter out zero-width
        .filter(|x| x.1 > 0)
        .inspect(|x| {
            calculated_width += x.1;
            all_unit_len &= x.0.len() == 1;
        })
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

    if return_all_null || output_height == 0 {
        let arr =
            FixedSizeListArray::new_null(dtype.to_arrow(CompatLevel::newest()), output_height);
        return Ok(ArrayChunked::with_chunk(args[0].name().clone(), arr).into_column());
    }

    // Combine validities
    let outer_validity = validities.into_iter().fold(None, |a, b| {
        debug_assert_eq!(b.len(), output_height);
        combine_validities_and(a.as_ref(), Some(&b))
    });

    // At this point the output height and all arrays should have non-zero length
    let out = if all_unit_len && width > 0 {
        // Fast-path for all scalars
        let inner_arr = unsafe { horizontal_flatten_unchecked(&arrays, &widths, 1) };

        let arr = FixedSizeListArray::new(
            dtype.to_arrow(CompatLevel::newest()),
            1,
            inner_arr,
            outer_validity,
        );

        return Ok(ArrayChunked::with_chunk(args[0].name().clone(), arr)
            .into_column()
            .new_from_index(0, output_height));
    } else {
        let inner_arr = if width == 0 {
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
        ArrayChunked::with_chunk(args[0].name().clone(), arr).into_column()
    };

    Ok(out)
}
