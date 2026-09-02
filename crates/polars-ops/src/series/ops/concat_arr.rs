use arrow::compute::utils::combine_validities_and;
use polars_array::PlFixedSizeListArray;
use polars_array::arrow::{export, import};
use polars_compute::horizontal_flatten::horizontal_flatten_unchecked;
use polars_core::chunked_array::arrow_bridge::chunk_to_arrow;
use polars_core::prelude::{ArrayChunked, Column, DataType, IntoColumn, StaticArray};
use polars_core::series::Series;
use polars_error::{PolarsContext, PolarsResult};
use polars_utils::broadcast::broadcast_len;
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

    let output_height = broadcast_len(args.iter()).context("concat_arr")?;
    let mut calculated_width = 0;
    // If there is a `Array` column with a single NULL, the output will be entirely NULL.
    let mut return_all_null = false;
    // Indicates whether all `arrays` have unit length (excluding zero-width arrays)
    let mut all_unit_len = true;
    let mut validities = Vec::with_capacity(args.len());

    let (arrays, widths): (Vec<_>, Vec<_>) = args
        .iter()
        .map(|c| {
            let len = c.len();

            // Don't expand scalars to height, this is handled by the `horizontal_flatten` kernel.
            let s = c.as_materialized_series_maintain_scalar();

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

                    // TODO(polars-array-scalar): the flatten kernel is an Arrow one, so a scalar
                    // chunk is written out here rather than the one row it stands for being
                    // concatenated once.
                    (
                        chunk_to_arrow(arr.downcast_as_array()).values().clone(),
                        *width,
                    )
                },
                dtype => {
                    debug_assert_eq!(dtype, inner_dtype);
                    // Note: We ignore the validity of non-array input columns, their outer is always valid after
                    // being reshaped to (-1, 1).
                    (export::to_arrow(&*s.rechunk().into_chunks()[0]), 1)
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

    if return_all_null || output_height == 0 {
        return Ok(ArrayChunked::full_null_with_dtype(
            args[0].name().clone(),
            output_height,
            inner_dtype,
            width,
        )
        .into_column());
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

        let arr =
            PlFixedSizeListArray::new(import::from_arrow(&*inner_arr), width, 1, outer_validity);

        let mut out = ArrayChunked::with_chunk(args[0].name().clone(), arr);
        unsafe { out.to_logical(inner_dtype.clone()) };

        return Ok(out.into_column().new_from_index(0, output_height));
    } else {
        let inner_arr = if width == 0 {
            export::to_arrow(
                &*Series::new_empty(PlSmallStr::EMPTY, inner_dtype)
                    .into_chunks()
                    .into_iter()
                    .next()
                    .unwrap(),
            )
        } else {
            unsafe { horizontal_flatten_unchecked(&arrays, &widths, output_height) }
        };

        let arr = PlFixedSizeListArray::new(
            import::from_arrow(&*inner_arr),
            width,
            output_height,
            outer_validity,
        );

        let mut out = ArrayChunked::with_chunk(args[0].name().clone(), arr);
        unsafe { out.to_logical(inner_dtype.clone()) };

        out.into_column()
    };

    Ok(out)
}
